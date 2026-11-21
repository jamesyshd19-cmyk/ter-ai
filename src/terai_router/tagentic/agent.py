"""The Tagentic1 loop: short thinking line + tool call per step, then a separate final answer.

Each model turn returns one JSON object (forced by a JSON schema on llama-cpp-python)::

    {"think": "อ่าน config ก่อน", "tool": "read_file", "args": {"path": "pyproject.toml"}}
    {"think": "ได้คำตอบแล้ว", "tool": "final", "answer": "..."}

The tool runs, the code writes a short note on what came back, and the full (clipped) output goes back to the model
as the next message. The run stops at the final answer, the step limit or the time limit (then the model is asked for
its final answer). Every reply the user reads is the model's own text, passed through unchanged: there are no fixed
replies and no output guard. Identity is set by Systemprompt.md only.

A session keeps the conversation (user lines and final answers) in ``history``. The first turn of a session is the
model's own greeting, asked for with the ``[session_start]`` event described in Systemprompt.md.

Verbatim tasks (issue #18): when the task gives a line-by-line specification with word and character counts
("Line 6 (23 words, 156 characters): ..."), the final answer is checked line by line (``verbatim.check``). A line
that differs is sent back to the model with what is wrong, and the model writes that line again (at most
``VERBATIM_ROUNDS`` rounds, with a short system message that holds only that line's specification; where the
specification fixes every word, the words are listed whole, with "built by joining `a` + `b`" already read as ``ab``, and the model answers that line as a list of its words, which the code joins with one
space each). Every character of every word is the model's own; the check never edits a word, so a line the model
still gets wrong stays as the model wrote it. The check shows in the trace as tool
``verbatim_check``.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import memory as M
from . import verbatim as V
from .context import workspace_context
from .tools import PendingAction, Toolbox
from .trace import NOTE_MAX, THINK_MAX, Step, Trace, shorten

OBS_MAX = 2500          # characters of the newest tool output sent to the model
OBS_OLD = 400           # older outputs are cut to this when the history grows
HISTORY_BUDGET = 9000   # characters of step history before older outputs are cut
STEP_TOKENS = 700
SESSION_START = '[session_start]'   # event message (model input only) that opens a session
VERBATIM_TOKENS = 2048  # tokens for a step when the task is a verbatim line specification (the whole text is one answer)
VERBATIM_ROUNDS = 4     # rounds of line rewrites after the verbatim check
VERBATIM_MAX_LINES = 40  # more wrong lines than this: no line rewrites


def step_schema(tools: list) -> dict:
    return {
        'type': 'object',
        'properties': {
            'think': {'type': 'string'},
            'tool': {'type': 'string', 'enum': list(tools) + ['final']},
            'args': {'type': 'object'},
            'answer': {'type': 'string'},
        },
        'required': ['think', 'tool'],
    }


FINAL_SCHEMA = {
    'type': 'object',
    'properties': {'think': {'type': 'string'}, 'tool': {'type': 'string', 'enum': ['final']},
                   'answer': {'type': 'string'}},
    'required': ['think', 'tool', 'answer'],
}


def parse_step(raw: str) -> Optional[dict]:
    """The step object from a model reply (tolerates ```json fences and text around the object)."""
    s = (raw or '').strip()
    s = re.sub(r'^```(?:json)?\s*|\s*```$', '', s)
    start = s.find('{')
    if start < 0:
        return None
    dec = json.JSONDecoder()
    try:
        obj, _ = dec.raw_decode(s[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get('tool'), str):
        return None
    if not isinstance(obj.get('args'), dict):
        obj['args'] = {}
    obj['think'] = str(obj.get('think') or '')
    return obj


@dataclass
class RunResult:
    final: str
    trace: Trace
    stopped: str
    messages: list = field(default_factory=list)
    pending: Optional[dict] = None   # {tool, args, summary} when a gated call waits for the user (relay mode)
    state: Optional[dict] = None     # what resume() needs to go on after --approve / --deny


class Tagentic:
    def __init__(self, lm, toolbox: Toolbox, home=None, max_steps: int = 12, timeout: float = 600.0,
                 seed: Optional[int] = 0, temperature: float = 0.2,
                 on_step: Optional[Callable[[Step], None]] = None, files: Optional[list] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.lm = lm
        self.tools = toolbox
        self.home = home if home is not None else toolbox.home
        self.max_steps = max_steps
        self.timeout = timeout
        self.seed = seed
        self.temperature = temperature
        self.on_step = on_step
        self.files = files
        self.clock = clock
        self.last: Optional[RunResult] = None
        self._tokens = STEP_TOKENS

    def system_prompt(self) -> str:
        public = bool(getattr(self.tools, 'public', False))
        files = self.files if self.files is not None else M.load_memory(self.home, public=public)
        if public:   # drop owner annotations such as "(Bogus 2026-11-06)" from the shared prompt files
            return (re.sub(r'\s*\(Bogus \d{4}-\d{2}-\d{2}\)', '', M.system_text(files))
                    + '\n\n===== public chat =====\n'
                    + 'This chat is a member of the public on Telegram, not the owner. You have no User.md, no Memory.md '
                    + 'and no workspace here: file, code, shell and remember tools are off. Do not guess who the user is. '
                    + 'Answer general questions from Knowledge.md first, then from your own knowledge. '
                    + 'If the user gives you information to add, call add_knowledge (it waits for a fact check).\n'
                    + 'tools (exact names and arguments):\n' + self.tools.describe())
        return (M.system_text(files)
                + '\n\n' + workspace_context(self.tools.ws)
                + f'\n\n===== workspace =====\nroot: {self.tools.ws.root}\n'
                + 'tools (exact names and arguments):\n' + self.tools.describe())

    # ----------------------------------------------------------------------------------------------- model
    def _ask(self, messages: list, schema: dict) -> str:
        # stop_on_json: generation ends at the closing brace of the step object (no trailing tokens)
        return self.lm.chat(messages, max_tokens=self._tokens, temperature=self.temperature, seed=self.seed,
                            json_schema=schema, stop_on_json=True)

    @staticmethod
    def _compress(messages: list, base: int) -> None:
        """Cut older tool outputs of this turn when its step history gets long (system, history, task stay whole)."""
        total = sum(len(m['content']) for m in messages[base:])
        if total <= HISTORY_BUDGET:
            return
        for m in messages[base:-1]:
            if m['role'] == 'user' and m['content'].startswith('ผลจาก') and len(m['content']) > OBS_OLD:
                m['content'] = m['content'][:OBS_OLD] + '\n... (older output cut)'

    def _final(self, messages: list, task: str, note: str) -> tuple:
        """Ask for the final answer only (used at the step or time limit, and for the identity retry)."""
        messages.append({'role': 'user', 'content': note})
        raw = self._ask(messages, FINAL_SCHEMA)
        obj = parse_step(raw)
        if obj is None:
            text = re.sub(r'^```\w*\s*|\s*```$', '', (raw or '').strip()).strip()
            return ('' if text.startswith('{') else text), {}
        return str(obj.get('answer') or obj.get('think') or '').strip(), obj

    # ----------------------------------------------------------------------------------------------- run
    def run(self, task: str, history: Optional[list] = None) -> RunResult:
        """One turn. ``history`` is the earlier conversation of a session ([{role, content}, ...])."""
        messages = ([{'role': 'system', 'content': self.system_prompt()}] + list(history or [])
                    + [{'role': 'user', 'content': task}])
        return self._loop(task, messages, Trace(), len(messages), self.clock())

    def resume(self, state: dict, approved: bool) -> RunResult:
        """Go on with a turn that stopped at a gated call (relay mode): run it when approved, else tell the model no."""
        t0 = self.clock() - float(state.get('elapsed') or 0.0)
        messages = [{'role': 'system', 'content': self.system_prompt()}] + list(state['messages'])
        trace = Trace()
        for d in state.get('steps') or []:
            trace.add(Step(**d))
        p = state['pending']
        self._do_step(p['think'], p['tool'], p['args'], trace, messages, int(state['base']), allow=approved)
        return self._loop(state['task'], messages, trace, int(state['base']), t0)

    def _do_step(self, think, tool, args, trace, messages, base, allow=None) -> None:
        s0 = time.monotonic()
        res = self.tools.call(tool, args, allow=allow)
        step = trace.add(Step(step=len(trace.steps) + 1, think=think, tool=tool, args=args, ok=res.ok,
                              note=shorten(res.note, NOTE_MAX), ms=int((time.monotonic() - s0) * 1000)))
        if self.on_step:
            self.on_step(step)
        messages.append({'role': 'assistant', 'content': json.dumps(
            {'think': think, 'tool': tool, 'args': args}, ensure_ascii=False)})
        out = res.output if len(res.output) <= OBS_MAX else res.output[:OBS_MAX] + '\n... (cut)'
        messages.append({'role': 'user', 'content': f'ผลจาก {tool} ({"ok" if res.ok else "error"}):\n{out}'})
        self._compress(messages, base)

    def _verbatim_pass(self, answer: str, spec: 'V.Spec', messages: list, trace: Trace) -> str:
        """Check a verbatim answer line by line; ask the model to write each wrong line again (issue #18)."""
        for rnd in range(VERBATIM_ROUNDS + 1):
            issues = V.check(answer, spec)
            want = sum(1 for _ in spec.lines)
            ok = want - len(issues)
            if not issues:
                note = f'{ok}/{want} lines match the specification'
            elif rnd == VERBATIM_ROUNDS or len(issues) > VERBATIM_MAX_LINES:
                note = (f'{ok}/{want} lines match; still different: '
                        + ', '.join(f'line {i.line}' for i in issues) + ' (kept as the model wrote them)')
            else:
                note = f'{ok}/{want} lines match; rewrite ' + ', '.join(f'line {i.line}' for i in issues)
            step = trace.add(Step(step=len(trace.steps) + 1, think='เช็กคำตอบกับ spec ทีละบรรทัด',
                                  tool='verbatim_check', args={'round': rnd + 1}, ok=not issues,
                                  note=shorten(note, NOTE_MAX), ms=0))
            if self.on_step:
                self.on_step(step)
            if not issues or rnd == VERBATIM_ROUNDS or len(issues) > VERBATIM_MAX_LINES:
                return answer
            rows = answer.split('\n')
            if spec.total:
                rows = rows[:spec.total]   # the specification states the line count: lines past it are dropped
            while len(rows) < max(spec.lines):
                rows.append('')
            for issue in issues:
                if issue.line > len(rows):
                    continue
                if spec.lines[issue.line].blank:
                    rows[issue.line - 1] = ''   # a specified blank line: nothing for the model to write
                    continue
                line = rows[issue.line - 1]
                if V.repair_mode(spec.lines[issue.line], issue, line):
                    rows[issue.line - 1] = self._repair_words(issue, line, rnd)
                    continue
                # a short system message: the rewrite needs only this line and its specification
                ask = [{'role': 'system', 'content': V.REWRITE_SYSTEM},
                       {'role': 'user', 'content': V.line_hint(issue, spec, rows[issue.line - 1])}]
                seed = self.seed
                if seed is not None:
                    self.seed = seed + rnd   # a new seed each round: the same prompt and seed give the same line
                try:
                    raw = self._ask(ask, V.rewrite_schema(spec.lines[issue.line], FINAL_SCHEMA))
                finally:
                    self.seed = seed
                new = V.read_rewrite(parse_step(raw), spec.lines[issue.line])
                if new and '\n' not in new:
                    rows[issue.line - 1] = new
            answer = '\n'.join(rows)
        return answer

    def _repair_words(self, issue: 'V.Issue', line: str, rnd: int) -> str:
        """Ask the model to write each wrong word of a line again, in plain text (no JSON: a word's quote needs no
        escape, issue #18), and put the words it wrote back at their positions."""
        got = {}
        seed = self.seed
        for pos, want, wrong in issue.words:
            ask = [{'role': 'system', 'content': V.REPAIR_SYSTEM},
                   {'role': 'user', 'content': V.word_hint(issue.line, pos, want, wrong)}]
            if seed is not None:
                self.seed = seed + rnd
            try:
                raw = self.lm.chat(ask, max_tokens=32 + 4 * len(want), temperature=self.temperature, seed=self.seed)
            finally:
                self.seed = seed
            word = V.read_word(raw)
            if word:
                got[pos] = word
        return V.put_words(line, got)

    def _loop(self, task: str, messages: list, trace: Trace, base: int, t0: float) -> RunResult:
        schema = step_schema(self.tools.names)
        spec = V.parse_spec(task)
        self._tokens = VERBATIM_TOKENS if spec else STEP_TOKENS
        stopped, answer, bad_json = '', '', 0
        while True:
            if self.clock() - t0 > self.timeout:
                stopped = 'timeout'
                break
            if len(trace.steps) >= self.max_steps:
                stopped = 'max_steps'
                break
            raw = self._ask(messages, schema)
            obj = parse_step(raw)
            if obj is None:
                text = re.sub(r'^```\w*\s*|\s*```$', '', (raw or '').strip()).strip()
                if text and not text.startswith('{'):
                    answer, stopped = text, 'answer'   # the model answered in plain text: that is its answer
                    break
                bad_json += 1
                if bad_json > 1:
                    stopped = 'error'
                    answer = text
                    break
                messages.append({'role': 'assistant', 'content': (raw or '')[:500]})
                messages.append({'role': 'user', 'content': 'ตอบเป็น JSON object เดียวตามรูปแบบ step: '
                                                        '{"think": "...", "tool": "...", "args": {...}}'})
                continue
            think = shorten(obj['think'], THINK_MAX) or '-'   # trace line only; the answer is never shortened
            tool, args = obj['tool'], obj['args']
            if tool == 'final':
                answer = str(obj.get('answer') or '').strip()
                if not answer:
                    answer, _ = self._final(messages, task, 'ใส่คำตอบใน "answer" ตอบเป็น JSON tool="final"')
                stopped = 'answer'
                messages.append({'role': 'assistant', 'content': json.dumps(obj, ensure_ascii=False)})
                break
            try:
                self._do_step(think, tool, args, trace, messages, base)
            except PendingAction as p:   # relay mode: stop here, the next command approves or denies
                pending = {'tool': p.tool, 'args': p.call_args, 'summary': p.summary}
                state = {'task': task, 'messages': messages[1:], 'base': base,
                         'steps': [s.to_dict() for s in trace.steps], 'elapsed': self.clock() - t0,
                         'pending': dict(pending, think=think)}
                trace.stopped, trace.seconds = 'pending', self.clock() - t0
                self.last = RunResult(final='', trace=trace, stopped='pending', messages=messages,
                                      pending=pending, state=state)
                return self.last
        if stopped in ('max_steps', 'timeout'):
            why = f'ครบ {self.max_steps} steps' if stopped == 'max_steps' else f'ครบเวลา {self.timeout:g} วินาที'
            answer, _ = self._final(messages, task, f'หยุดใช้ tool แล้ว ({why}) บอกผู้ใช้ว่างานหยุดเพราะอะไร '
                                                    'สรุปสิ่งที่ทำได้และสิ่งที่ยังไม่เสร็จจากผล tool ที่มี ตอบเป็น JSON tool="final"')
        if spec and stopped == 'answer' and answer:
            answer = self._verbatim_pass(answer, spec, messages, trace)
            if messages and messages[-1]['role'] == 'assistant':
                messages[-1] = {'role': 'assistant', 'content': json.dumps(
                    {'think': 'verbatim', 'tool': 'final', 'answer': answer}, ensure_ascii=False)}
        trace.final, trace.stopped = answer, stopped
        trace.seconds = self.clock() - t0
        self.last = RunResult(final=answer, trace=trace, stopped=stopped, messages=messages)
        return self.last

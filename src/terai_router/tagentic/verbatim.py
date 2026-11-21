# SPDX-License-Identifier: LicenseRef-TER-AI-1.0
"""Line check for verbatim (copy / text-return) tasks.

Some tasks give a line-by-line specification of a text the model has to write out exactly, for example::

    Line 6 (23 words, 156 characters): ...
      - Word 10: built by joining `ระหว่าง` + `แก้`
      - Word 11: `ชื่อ`
    Line 9: blank.

Thai has no spaces between words, so a small model can merge two specified words into one (issue #18: it wrote
``ระหว่างแก้ชื่อ`` for the two words ``ระหว่างแก้`` and ``ชื่อ``). ``parse_spec`` reads the per-line word and
character counts (and the exact words where the specification fixes them), ``check`` compares an answer with them,
and ``line_hint`` writes the note that asks the model to write a wrong line again. The check never edits the answer
itself: the agent only asks the model again and keeps the model's own text (see ``Tagentic._verbatim_pass``).

Counts follow the specification's own notation: a word is anything separated by spaces (``str.split``) and a
character is one Unicode code point (``len``).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

_LINE = re.compile(r'^[ \t]*(?:#+[ \t]*)?Line[ \t]+(\d+)[ \t]*'
                   r'(?:\((\d+)[ \t]+words?,[ \t]*(\d+)[ \t]+char(?:acter)?s?\))?[ \t]*:[ \t]*(.*)$',
                   re.M | re.I)
_WORD = re.compile(r'^[ \t]*[-*][ \t]*Words?[ \t]+(\d+)(?:[ \t]*[-\u2013][ \t]*(\d+))?[ \t]*:[ \t]*(.*)$', re.I)
_TICK = re.compile(r'`([^`]*)`')
_JOIN = re.compile(r'^built by joining[ \t]+((?:`[^`]*`[ \t]*\+[ \t]*)*`[^`]*`)', re.I)
_TOTAL = re.compile(r'\b(\d+)[ \t]+lines\b', re.I)


@dataclass
class LineSpec:
    number: int
    blank: bool = False
    words: Optional[int] = None
    chars: Optional[int] = None
    expected: dict = field(default_factory=dict)   # word position (1-based) -> exact word, where the spec fixes it
    text: str = ''                                 # the specification block of this line, as written


@dataclass
class Spec:
    lines: dict                       # line number -> LineSpec
    total: Optional[int] = None       # total line count, when the specification states it


@dataclass
class Issue:
    line: int
    want_words: Optional[int]
    got_words: int
    want_chars: Optional[int]
    got_chars: int
    blank: bool = False
    word: Optional[int] = None        # first position whose fixed word differs
    want_word: str = ''
    got_word: str = ''
    words: list = field(default_factory=list)   # every (position, want, got) whose fixed word differs


def _expected_words(first: int, last: int, rest: str) -> dict:
    rest = rest.strip()
    m = _JOIN.match(rest)
    if m and first == last:
        return {first: ''.join(_TICK.findall(m.group(1)))}
    want = last - first + 1
    ticks = _TICK.findall(rest)
    if rest.startswith('`') and ticks:
        parts = ticks[0].split(' ')
        if len(parts) == want and all(parts):
            return {first + i: p for i, p in enumerate(parts)}
    if want > 1 and len(ticks) > 1:
        # "It opens with `"Wabi:` and continues with `The AI Assistant` and ends with `You Use Together"`"
        parts = ' '.join(ticks).split(' ')
        if len(parts) == want and all(parts):
            return {first + i: p for i, p in enumerate(parts)}
    return {}


def parse_spec(text: str) -> Optional[Spec]:
    """The line specification in ``text``, or None when it has no per-line counts (not a verbatim task)."""
    if not text:
        return None
    heads = list(_LINE.finditer(text))
    lines = {}
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        block = text[h.start():end]
        # the block stops at the next Markdown heading (for example "## Notes on the source")
        cut = re.search(r'^[ \t]*#{1,6}[ \t]', block[1:], re.M)
        if cut:
            block = block[:cut.start() + 1]
        n = int(h.group(1))
        rest = h.group(4).strip()
        ls = LineSpec(number=n, text=block.rstrip())
        if h.group(2):
            ls.words, ls.chars = int(h.group(2)), int(h.group(3))
        elif re.match(r'^(?:blank|empty)\b', rest, re.I):
            ls.blank, ls.words, ls.chars = True, 0, 0
        else:
            continue
        for row in block.splitlines()[1:]:
            w = _WORD.match(row)
            if w:
                first = int(w.group(1))
                last = int(w.group(2) or first)
                ls.expected.update(_expected_words(first, last, w.group(3)))
        lines[n] = ls
    if not lines or not any(not ls.blank for ls in lines.values()):
        return None
    total = None
    for m in _TOTAL.finditer(text):
        total = int(m.group(1))
        break
    return Spec(lines=lines, total=total)


def check_line(ls: LineSpec, line: str) -> Optional[Issue]:
    words = line.split()
    bad_count = (ls.words is not None and len(words) != ls.words) or (ls.chars is not None and len(line) != ls.chars)
    issue = Issue(line=ls.number, want_words=ls.words, got_words=len(words), want_chars=ls.chars,
                  got_chars=len(line), blank=ls.blank)
    for pos in sorted(ls.expected):
        got = words[pos - 1] if pos - 1 < len(words) else ''
        if got != ls.expected[pos]:
            issue.words.append((pos, ls.expected[pos], got))
    if issue.words:
        issue.word, issue.want_word, issue.got_word = issue.words[0]
        return issue
    return issue if bad_count else None


def check(answer: str, spec: Spec) -> list:
    """Issues of ``answer`` against ``spec``: one per specified line that differs (missing lines count as empty),
    and one per line past the total line count the specification states."""
    rows = (answer or '').split('\n')
    out = []
    for n in sorted(spec.lines):
        line = rows[n - 1] if n - 1 < len(rows) else ''
        issue = check_line(spec.lines[n], line)
        if issue:
            out.append(issue)
    if spec.total and len(rows) > spec.total:
        for n in range(spec.total + 1, len(rows) + 1):   # lines past the stated total
            line = rows[n - 1]
            out.append(Issue(line=n, want_words=0, got_words=len(line.split()), want_chars=0, got_chars=len(line),
                             blank=True))
    return out


def line_hint(issue: Issue, spec: Spec, line: str) -> str:
    """The note that asks the model to write one line again (sent as a user message, never shown to the user)."""
    ls = spec.lines[issue.line]
    why = []
    if ls.blank:
        why.append('this line must be empty')
    else:
        if issue.want_words is not None and issue.got_words != issue.want_words:
            why.append(f'it has {issue.got_words} words but the specification says {issue.want_words}')
        if issue.want_chars is not None and issue.got_chars != issue.want_chars:
            why.append(f'it has {issue.got_chars} characters but the specification says {issue.want_chars}')
        for pos, want, got in issue.words[:6]:
            why.append(f'word {pos} must be exactly `{want}`, you wrote `{got}`')
    if ls.words and len(ls.expected) == ls.words:
        # every word is fixed by the specification: list them whole ("built by joining `a` + `b`" is the word ab)
        words = '\n'.join(f'{i}. {ls.expected[i]}' for i in range(1, ls.words + 1))
        spec_part = (f'The {ls.words} words of this line, in order (a joined word is already written whole here; '
                     f'do not split it):\n{words}')
    else:
        spec_part = f'Specification of this line:\n{ls.text}'
    target = ' '.join(ls.expected[i] for i in sorted(ls.expected)) if spec_part.startswith('The ') else ls.text
    close = difflib.SequenceMatcher(None, line, target, autojunk=False).ratio() >= 0.5
    wrote = f'You wrote:\n{line}\n\n' if close else ''   # a line far off (another line's text) is not shown again
    if not close:
        why = ['it is not this line of the text']
    if words_mode(ls):
        ask = ('Write line ' + str(issue.line) + f' again as a list of its {ls.words} words: put each word above, '
               'in this order, as one item of "words", copied character for character (keep every quote, '
               'bracket, colon, comma and dash; the line is these words joined by one space). A word with a double '
               'quote " keeps it: in JSON write it as \\" (for example "end.\\"" is the word end."). Do not write '
               '"built by joining", "+" or the word numbers. Answer as JSON tool="final" with the list in "words".')
    else:
        ask = ('Write line ' + str(issue.line) + ' again: the words in this order with exactly one space between '
               'every two words, including between Thai words, and no other spaces. Do not write the words "built '
               'by joining", "+" or the word numbers. Keep every character exactly. Answer as JSON tool="final" with '
               'only this one line in "answer".')
    return ('ตรวจคำตอบกับ spec แล้ว บรรทัดนี้ไม่ตรง ให้เขียนบรรทัดนี้ใหม่บรรทัดเดียว\n'
            f'Your line {issue.line} does not match the specification: ' + '; '.join(why) + '.\n'
            f'{wrote}{spec_part}\n\n' + ask)


def words_mode(ls: LineSpec) -> bool:
    """True when the specification fixes every word of the line: the rewrite then asks for a list of words."""
    return bool(ls.words) and len(ls.expected) == ls.words


def rewrite_schema(ls: LineSpec, final_schema: dict) -> dict:
    """JSON schema of a line rewrite: a list of exactly ``ls.words`` words, or the plain final step."""
    if not words_mode(ls):
        return final_schema
    return {'type': 'object',
            'properties': {'think': {'type': 'string'}, 'tool': {'type': 'string', 'enum': ['final']},
                           'words': {'type': 'array', 'items': {'type': 'string'},
                                     'minItems': ls.words, 'maxItems': ls.words}},
            'required': ['think', 'tool', 'words']}


def read_rewrite(obj: Optional[dict], ls: LineSpec) -> str:
    """The rewritten line from the model's rewrite step ('' when it gave none).

    In word-list mode the model writes every word itself and the line is its words joined by one space each: the
    spaces between words, which issue #18 lost, are set by the code; no character of a word is changed.
    """
    if not obj:
        return ''
    words = obj.get('words')
    if words_mode(ls) and isinstance(words, list) and words and all(isinstance(w, str) for w in words):
        return ' '.join(w.strip() for w in words)
    return str(obj.get('answer') or '').strip('\n')


REWRITE_SYSTEM = ('You rewrite one line of a text so that it matches its specification exactly. '
                  'Reply with one JSON object: {"think": "...", "tool": "final", "answer": "<the line>"}, or, when '
                  'asked for a list of words, {"think": "...", "tool": "final", "words": ["...", "..."]}.')

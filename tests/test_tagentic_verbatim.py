# SPDX-License-Identifier: LicenseRef-TER-AI-1.0
"""Regression tests for issue #18: Tagentic1 merged two Thai words in a verbatim text-return task.

The specification said line 6 has word 10 `ระหว่าง` + `แก้` and word 11 `ชื่อ`; the model wrote `ระหว่างแก้ชื่อ` (one space
less, 22 words / 155 characters instead of 23 / 156).
"""
import json

from terai_router.tagentic import verbatim as V
from terai_router.tagentic.agent import VERBATIM_ROUNDS, Tagentic
from terai_router.tagentic.tools import Toolbox
from terai_router.tagentic.workspace import Workspace

GOOD6 = 'ตอน อัปเดต playlist ระหว่างแก้ ชื่อ เพลง'
BAD6 = 'ตอน อัปเดต playlist ระหว่างแก้ชื่อ เพลง'
LINE1 = 'Linktree report'

SPEC = f"""Write the text below exactly. It has 3 lines.

## Lines

Line 1 ({len(LINE1.split())} words, {len(LINE1)} characters): a short title.
  - Words 1-2: `{LINE1}`

Line 2: blank.

Line 3 ({len(GOOD6.split())} words, {len(GOOD6)} characters): what changed.
  - Word 1: `ตอน`
  - Word 2: `อัปเดต`
  - Word 3: `playlist`
  - Word 4: built by joining `ระหว่าง` + `แก้`
  - Word 5: `ชื่อ`
  - Word 6: `เพลง`

## Notes

Keep every space.
"""
GOOD = f'{LINE1}\n\n{GOOD6}'
BAD = f'{LINE1}\n\n{BAD6}'


def test_parse_spec_reads_counts_and_words():
    spec = V.parse_spec(SPEC)
    assert spec.total == 3
    assert spec.lines[2].blank
    l3 = spec.lines[3]
    assert (l3.words, l3.chars) == (6, len(GOOD6))
    assert l3.expected[4] == 'ระหว่างแก้' and l3.expected[5] == 'ชื่อ'
    assert spec.lines[1].expected == {1: 'Linktree', 2: 'report'}
    assert 'Notes' not in l3.text


def test_not_a_spec():
    assert V.parse_spec('ช่วยสรุปไฟล์ README.md') is None
    assert V.parse_spec('') is None


def test_check_finds_the_merged_thai_space():
    spec = V.parse_spec(SPEC)
    assert V.check(GOOD, spec) == []
    issues = V.check(BAD, spec)
    assert len(issues) == 1
    i = issues[0]
    assert i.line == 3 and (i.got_words, i.want_words) == (5, 6) and i.got_chars == i.want_chars - 1
    assert (i.word, i.want_word, i.got_word) == (4, 'ระหว่างแก้', 'ระหว่างแก้ชื่อ')
    hint = V.line_hint(i, spec, BAD6)
    assert 'line 3' in hint and 'ระหว่างแก้' in hint and 'tool="final"' in hint


def test_lines_past_total_are_issues():
    spec = V.parse_spec(SPEC)
    assert [i.line for i in V.check(GOOD + '\nextra', spec)] == [4]


def test_spec_opens_continues_ends_form():
    spec = V.parse_spec('It has 1 lines.\nLine 1 (3 words, 13 characters): a title.\n'
                        '  - Words 1-3: a title. It opens with `"Wabi:` and continues with `The` and ends with `AI"`\n')
    assert spec.lines[1].expected == {1: '"Wabi:', 2: 'The', 3: 'AI"'}


def test_blank_line_must_be_empty():
    spec = V.parse_spec(SPEC)
    issues = V.check(f'{LINE1}\nx\n{GOOD6}', spec)
    assert [i.line for i in issues] == [2]


class ScriptLM:
    """Fake model: returns the scripted steps in order and records what it was asked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append({'messages': messages, **kw})
        return self.replies.pop(0)


def _final(text):
    return json.dumps({'think': 'ตอบ', 'tool': 'final', 'args': {}, 'answer': text}, ensure_ascii=False)


def _agent(tmp_path, lm):
    ws, home = tmp_path / 'ws', tmp_path / 'home'
    ws.mkdir()
    home.mkdir()
    return Tagentic(lm, Toolbox(Workspace(ws), home=home, yes=True), home=home)


def test_agent_rewrites_the_wrong_line(tmp_path):
    lm = ScriptLM([_final(BAD), _final(GOOD6)])
    res = _agent(tmp_path, lm).run(SPEC)
    assert res.final == GOOD
    assert len(lm.calls) == 2
    assert 'ระหว่างแก้' in lm.calls[1]['messages'][-1]['content']
    checks = [s for s in res.trace.steps if s.tool == 'verbatim_check']
    assert [s.ok for s in checks] == [False, True]
    assert lm.calls[0]['max_tokens'] >= 2048


def test_agent_rewrite_as_word_list(tmp_path):
    words = json.dumps({'think': 'list', 'tool': 'final', 'words': GOOD6.split(' ')}, ensure_ascii=False)
    lm = ScriptLM([_final(BAD), words])
    res = _agent(tmp_path, lm).run(SPEC)
    assert res.final == GOOD
    schema = lm.calls[1]['json_schema']
    assert schema['properties']['words']['minItems'] == 6 == schema['properties']['words']['maxItems']


def test_agent_keeps_model_text_when_still_wrong(tmp_path):
    lm = ScriptLM([_final(BAD)] + [_final(BAD6)] * VERBATIM_ROUNDS)
    res = _agent(tmp_path, lm).run(SPEC)
    assert res.final == BAD           # the check never edits the answer itself
    checks = [s for s in res.trace.steps if s.tool == 'verbatim_check']
    assert len(checks) == VERBATIM_ROUNDS + 1 and not checks[-1].ok and 'line 3' in checks[-1].note
    seeds = [c['seed'] for c in lm.calls[1:]]
    assert len(set(seeds)) == VERBATIM_ROUNDS   # a new seed each rewrite round


def test_agent_no_check_for_normal_task(tmp_path):
    lm = ScriptLM([_final('สวัสดีครับ')])
    res = _agent(tmp_path, lm).run('สวัสดี')
    assert res.final == 'สวัสดีครับ'
    assert not [s for s in res.trace.steps if s.tool == 'verbatim_check']
    assert lm.calls[0]['max_tokens'] < 2048



# ---- issue #18, second slip: a quote attached to the last word of a quoted group ("... (7 videos)") was lost ----
QL = '- ผมแก้กลับเป็น "AI music playlist (7 videos)" แล้ว'
QL_BAD = QL.replace('videos)"', 'videos)')
QSPEC = f"""Write the text below exactly. It has 1 lines.

Line 1 ({len(QL.split())} words, {len(QL)} characters): the name was changed back.
  - Word 1: `-`
  - Word 2: built by joining `ผม` + `แก้` + `กลับ` + `เป็น`
  - Words 3-7: `"AI music playlist (7 videos)"` (closing parenthesis then closing quote)
  - Word 8: `แล้ว`
"""


def test_spec_keeps_quote_and_bracket_on_words():
    ls = V.parse_spec(QSPEC).lines[1]
    assert ls.expected[3] == '"AI' and ls.expected[7] == 'videos)"' and V.words_mode(ls)


def test_check_flags_lost_trailing_quote():
    spec = V.parse_spec(QSPEC)
    assert V.check(QL, spec) == []
    (i,) = V.check(QL_BAD, spec)
    assert i.words == [(7, 'videos)"', 'videos)')] and i.got_chars == i.want_chars - 1
    assert V.repair_mode(spec.lines[1], i, QL_BAD)


def test_read_rewrite_keeps_escaped_quote():
    ls = V.parse_spec(QSPEC).lines[1]
    obj = json.loads(json.dumps({'think': 't', 'tool': 'final', 'words': QL.split(' ')}, ensure_ascii=False))
    assert V.read_rewrite(obj, ls) == QL


def test_read_word_and_put_words():
    assert V.read_word('v\ni\nd\ne\no\ns\n)\n"') == 'videos)"'   # one character per line
    assert V.read_word('v\ni\nd\ne\no\ns\n)\n') == 'videos)'    # the model's characters only: no quote added
    assert V.read_word('\u27e6videos)"\u27e7') == 'videos)"'
    assert V.read_word('word: \u27e6เป็น,\u27e7 done') == 'เป็น,'
    assert V.read_word('videos)"') == ''              # no brackets: nothing taken
    assert V.read_word('\u27e6two words\u27e7') == ''  # not one word
    assert V.read_word('\u27e6\u27e7') == ''
    assert V.put_words(QL_BAD, {7: 'videos)"'}) == QL
    assert V.put_words('a  b', {}) == 'a b'


def test_agent_repairs_trailing_quote_word_in_plain_text(tmp_path):
    lm = ScriptLM([_final(QL_BAD), 'v\ni\nd\ne\no\ns\n)\n"'])
    res = _agent(tmp_path, lm).run(QSPEC)
    assert res.final == QL
    repair = lm.calls[1]
    assert repair.get('json_schema') is None                 # plain text: the quote needs no JSON escape
    note = repair['messages'][-1]['content']
    assert note.endswith('v\ni\nd\ne\no\ns\n)   [closing parenthesis]\n"   [straight double quote]')
    assert 'videos)\n' not in note.split('\n', 1)[0]           # what the model wrote before is not shown


def test_agent_repair_never_adds_the_quote_itself(tmp_path):
    lm = ScriptLM([_final(QL_BAD)] + ['\u27e6videos)\u27e7'] * VERBATIM_ROUNDS)
    res = _agent(tmp_path, lm).run(QSPEC)
    assert res.final == QL_BAD                               # the model never wrote the quote: the code adds none
    checks = [s for s in res.trace.steps if s.tool == 'verbatim_check']
    assert len(checks) == VERBATIM_ROUNDS + 1 and not checks[-1].ok


def test_agent_repairs_after_word_list_rewrite(tmp_path):
    words = json.dumps({'think': 'l', 'tool': 'final', 'words': QL_BAD.split(' ')}, ensure_ascii=False)
    short = QL_BAD.replace(' แล้ว', 'แล้ว')                      # one word short: full word-list rewrite first
    lm = ScriptLM([_final(short), words, '\u27e6videos)"\u27e7'])
    res = _agent(tmp_path, lm).run(QSPEC)
    assert res.final == QL
    assert lm.calls[1]['json_schema']['properties']['words']['minItems'] == 8
    assert lm.calls[2].get('json_schema') is None

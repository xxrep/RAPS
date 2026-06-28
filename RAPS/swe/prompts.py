"""All RAPS-on-SWE-bench prompts + the injected edit tool, in one place (English).

Designed against failure modes observed on the dev set:
  - `sed -i`/`echo` edits corrupting multi-line Python (the dominant cause of wrong patches)
        -> APPLY_EDIT_TOOL (robust SEARCH/REPLACE applier) + SWE_RULES mandate
  - mocked/bogus reproductions that pass on broken code   -> Reproducer guidance (team.py)
  - edits in tests / site-packages / via git commit       -> SWE_RULES
  - routing noise from verbose needs                       -> BROKER_SYSTEM (short imperative)
Iterate prompts here without touching coordination logic in agent.py.
"""

EDIT_TOOL_PATH = "/tmp/apply_edit.py"
REPRO_PATH = "/tmp/raps_repro.py"

# A robust SEARCH/REPLACE file editor injected into the container at startup. It (1) requires the
# SEARCH text to match exactly once, and (2) refuses to write if the result is invalid Python —
# eliminating the sed/echo corruption that produced one-line garbage patches.
APPLY_EDIT_TOOL = r'''
import sys, re, ast
if len(sys.argv) < 2:
    sys.exit("ERROR: usage: python3 /tmp/apply_edit.py <file>  (SEARCH/REPLACE block on stdin)")
path = sys.argv[1]
data = sys.stdin.read()
# Tolerant parse: split on the SEARCH marker; for each block, SEARCH is before '=======',
# REPLACE is after it up to an optional '>>>>>>> REPLACE' or the end (heredoc EOF). The
# '>>>>>>> REPLACE' line is OPTIONAL (models often terminate with the heredoc delimiter).
edits = []
for chunk in data.split("<<<<<<< SEARCH")[1:]:
    if "=======" not in chunk:
        continue
    search, rest = chunk.split("=======", 1)
    rest = re.split(r"^>>>>>>>.*$", rest, maxsplit=1, flags=re.M)[0]
    edits.append((search.strip("\n"), rest.strip("\n")))
if not edits:
    sys.exit("ERROR: no block found. Format: <<<<<<< SEARCH (newline) old lines (newline) ======= (newline) new lines")
try:
    src = open(path).read()
except Exception as e:
    sys.exit(f"ERROR: cannot read {path}: {e}")
for search, replace in edits:
    if search == "":
        sys.exit("ERROR: empty SEARCH block")
    n = src.count(search)
    if n == 0:
        sys.exit("ERROR: SEARCH not found - copy the EXACT existing lines incl. indentation")
    if n > 1:
        sys.exit(f"ERROR: SEARCH matches {n} places - add surrounding lines to make it unique")
    src = src.replace(search, replace, 1)
if path.endswith(".py"):
    try:
        ast.parse(src)
    except SyntaxError as e:
        sys.exit(f"ERROR: edit would break Python syntax (line {e.lineno}: {e.msg}); NOT written")
open(path, "w").write(src)
print(f"OK: applied {len(edits)} edit block(s) to {path}")
'''

# Prepended to EVERY persona's system message. Hard, non-negotiable constraints.
SWE_RULES = """\
## TEAM HARD RULES (obey strictly)
- The repository under test lives at /testbed and is already installed in editable mode. Edit ONLY \
real source files under /testbed (e.g. /testbed/<package>/...). A change anywhere else does NOT count.
- NEVER edit tests (any path containing /tests/, test_*.py or *_test.py). NEVER edit files under \
site-packages, dist-packages, /opt/, or any installed copy of the package.
- NEVER run `git commit`, `git checkout`, `git reset`, `git stash`, or `git clean`.
- ## How to edit code (MANDATORY): use the robust editor, never `sed -i`/`echo`/`>>` for code logic \
(they corrupt multi-line Python). Apply a precise SEARCH/REPLACE:
  python3 {EDIT_TOOL_PATH} /testbed/<pkg>/<file>.py <<'EDIT'
  <<<<<<< SEARCH
  <copy the EXACT existing lines here, including indentation>
  =======
  <the replacement lines>
  >>>>>>> REPLACE
  EDIT
  The SEARCH text must match exactly once; the tool refuses edits that are non-unique or that break \
Python syntax. If it errors, read the message, widen the SEARCH context, and retry.
- After editing, run `git --no-pager diff` to confirm the change landed where intended.
- The team's canonical reproduction is {REPRO_PATH}. It MUST exit NONZERO while the bug is present and \
ZERO once fixed. The fix is ACCEPTED only when this script passes — you cannot submit before it does.
"""
SWE_RULES = SWE_RULES.replace("{EDIT_TOOL_PATH}", EDIT_TOOL_PATH).replace("{REPRO_PATH}", REPRO_PATH)

# BROKER: emits the single next NEED, embedding-matched to one specialist's subscription.
BROKER_SYSTEM = """\
You are the BROKER of a team fixing ONE software bug in a code repository.
Read the issue and the most recent command outputs, then decide the single most important NEXT need \
that moves the team toward a VERIFIED fix.

Choose the need that fits the current state:
- root cause / faulty file still unknown            -> a LOCATE need  ("locate the function that ...")
- cause known but bug not yet reproduced            -> a REPRODUCE need ("write a script reproducing ...")
- reproduced and located, but no fix applied yet    -> an EDIT need    ("edit <file> to ...")
- a source edit exists but is not yet verified      -> a VERIFY need   ("run the reproduction and tests")
- the fix passed and only a final check remains     -> a REVIEW need   ("review the diff for regressions")

Output ONLY the need as ONE short imperative phrase (<= 12 words). \
No 'THOUGHT', no code, no explanation, no quotes.
"""

# REACTIVE SUBSCRIPTION: specialize the routed persona's standing role to the current issue/state.
REFINE_SYSTEM = """\
You adapt a team member's role to the current step of fixing ONE software bug (reactive subscription).
Given the member's base role, the current need, and recent progress, rewrite the role into 2-3 concrete \
sentences stating exactly what this member should focus on RIGHT NOW — name the specific file, function, \
or symptom in play. Keep the member's core function intact. Do NOT solve the bug, write code, or state a \
final answer. Output only the rewritten role.
"""

# Shown when a persona's publication is not exactly one action block (format repair).
FORMAT_REMINDER = (
    "Your reply must contain EXACTLY ONE ```mswea_bash_command``` block with a single command "
    "(chain steps with && if needed, or use a here-doc). Re-issue your action now in that exact format."
)

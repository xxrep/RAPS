"""The SWE agent team (RAPS hosts).

Each persona declares a `subscription` (its standing intent). Every round the broker emits the
next-step NEED; `text-embedding-3-small` matches that NEED against these subscriptions to route
control. The selected persona then reactively specializes its `guidance` to the concrete issue
before publishing one action.

Subscriptions use distinct, action-oriented vocabulary so the embedding match is unambiguous
(verified: clean needs route locate->Localizer, edit->Editor, test->Verifier, etc.). Guidance
strings are written to counter the observed failure modes (wrong-location edits, premature submit).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    subscription: str   # matched against the broker's NEED via embeddings
    guidance: str       # role specialization, seeds the reactive-subscription rewrite


DEFAULT_TEAM = [
    Persona(
        name="Localizer",
        subscription=("locate the root-cause file and function; search the codebase with grep, find "
                      "and ripgrep; trace symbols, definitions, imports and call sites; read the "
                      "suspect source to understand it"),
        guidance=("You are the Localizer. THIS step: pinpoint the exact /testbed source file and "
                  "function responsible. `grep -rn` the issue's DISTINCTIVE identifiers across "
                  "/testbed — exact error-message text, function/class/attribute names, key symbols — "
                  "then read the top candidates with `sed -n`. ALSO grep the EXISTING tests for the "
                  "same class/function and read the most relevant ones to learn the expected "
                  "contract (what behaviour the fix must satisfy / not regress). Do NOT edit. Finish "
                  "by naming the precise /testbed file path and function/lines to change, and why."),
    ),
    Persona(
        name="Reproducer",
        subscription=("write a minimal standalone script that reproduces the reported bug and run it "
                      "to confirm the failure before any fix; establish the failing baseline"),
        guidance=("You are the Reproducer. THIS step: write the canonical reproduction to "
                  "/tmp/raps_repro.py, based on the EXACT code example in the issue text (use its "
                  "inputs and its stated expected output). Call the REAL public API (do NOT "
                  "mock/patch/stub the function or class under test — a mocked test passes on broken "
                  "code and is useless) and ASSERT the issue's expected behaviour, so it EXITS NONZERO "
                  "now (bug present) and ZERO once fixed. Run `python /tmp/raps_repro.py` and confirm "
                  "it currently FAILS with the reported symptom. Do not edit library source."),
    ),
    Persona(
        name="Editor",
        subscription=("edit and modify the source code to implement the fix; change the faulty "
                      "function; apply the code patch to the located file"),
        guidance=("You are the Editor. THIS step: make the minimal correct change to the ROOT CAUSE "
                  "in the located /testbed source file (never a test, never site-packages). FIRST "
                  "`sed -n 'START,ENDp' <file>` to see the exact current lines, then apply the robust "
                  "editor: `python3 /tmp/apply_edit.py <file> <<'EDIT'` then a <<<<<<< SEARCH / "
                  "======= / >>>>>>> REPLACE block, copying those lines VERBATIM (exact whitespace) "
                  "into SEARCH. MODIFY the existing code in place — do NOT add a duplicate definition. "
                  "NEVER use sed/echo to write code. Then `git --no-pager diff` to confirm. If "
                  "apply_edit errors 'SEARCH not found', re-read the lines and copy them exactly."),
    ),
    Persona(
        name="Verifier",
        subscription=("run the reproduction script and the repository's existing relevant tests; "
                      "check pass/fail and regressions; confirm the fix works, then submit"),
        guidance=("You are the Verifier. THIS step: run the canonical reproduction "
                  "`python /tmp/raps_repro.py` (it exits 0 only when fixed) and the most relevant "
                  "existing tests. Report exactly what passed and failed. Follow the submission "
                  "protocol ONLY if the reproduction now PASSES (exit 0) and a real /testbed source "
                  "diff exists; otherwise state precisely what still fails so the team can iterate."),
    ),
    Persona(
        name="Reviewer",
        subscription=("review and critique the current diff for correctness, generality and "
                      "regressions; verify the fix is in the right source location and not in tests"),
        guidance=("You are the Reviewer. THIS step: inspect `git --no-pager diff` and the reasoning. "
                  "Confirm the fix targets the root cause in the correct /testbed source file (not a "
                  "test, not site-packages), is general (not hard-coded to one case), and will not "
                  "regress. If anything is wrong, state exactly what must change next."),
    ),
]

# RAPS on SWE-bench — Handoff

This document hands off the **SWE-bench Lite** line of work: porting RAPS (the paper's
Reputation-Aware Publish-Subscribe coordination) onto an agentic code-repair benchmark, the
single-agent baselines it is measured against, and the (negative) result. It complements — does
not replace — the original `HANDOFF.md` (the QA / MMLU-GSM8K-HumanEval framework) and the design
note `RAPS_SWE_DESIGN.md`.

**TL;DR** — On SWE-bench Lite, RAPS multi-agent coordination gives **no resolved-rate gain** over a
single `mini-swe-agent` loop, on **both** `gpt-4o-mini` and `gpt-4o`. The real lever is the
backbone's single-step repair quality (10% → 20% resolved by swapping the model alone). The
framework is mechanically sound and fully instrumented; this is a clean, legitimate negative /
ceiling finding.

---

## 0. Results at a glance

All runs use the **same fixed 20-task stratified eval set** `runs/eval20.txt`. "Resolved" = official
`swebench.harness` verdict on the real `FAIL_TO_PASS` / `PASS_TO_PASS` tests.

| Run | Backbone | Resolved | Cost | Tokens | Avg rounds | Non-empty | Output dir |
|---|---|---|---|---|---|---|---|
| **Single-agent baseline** | gpt-4o-mini | **2/20 (10%)** | $0.78 | 5.02M | 22.9 | 16/20 | `runs/baseline_v2/` |
| **Single-agent baseline** | gpt-4o | **4/20 (20%)** | $20.81 | 8.11M | 33.0 | 15/20 | `runs/baseline_gpt4o/` |
| **RAPS** (full 20) | gpt-4o-mini | **1/20 (5%)** | $0.84 | 5.31M | 27.4 / 63.5 llm | 15/20 | `runs/raps_full20/` |
| **RAPS** (5-task slice) | gpt-4o | **2/5** | $3.89 | 1.48M | 29.0 / 67.2 llm | 5/5 | `runs/raps5_gpt4o/` |

Resolved instance IDs:
- baseline 4o-mini: `django-10914, django-11039`
- baseline gpt-4o: `django-10914, django-11001, django-11039, pytest-11143` (a strict **superset** of 4o-mini)
- RAPS 4o-mini full20: `django-11039` (it **lost** django-10914 — added a duplicate
  `FILE_UPLOAD_PERMISSIONS` line instead of editing the existing one → ineffective fix; gpt-4o-mini variance, not a gate bug)
- RAPS gpt-4o 5-task: `django-11001, pytest-11143`

### The head-to-head that matters (same 5 tasks, gpt-4o)

`runs/raps5.txt` = `django-11001, pytest-11143, sklearn-10508, matplotlib-22711, sympy-11400`
(2 the gpt-4o baseline solved + 2 it tried-but-failed + 1 it timed out on).

| | resolved | which | avg rounds | cost (same 5) |
|---|---|---|---|---|
| single-agent gpt-4o | 2/5 | django-11001, pytest-11143 | 35.6 | $5.84 |
| RAPS gpt-4o | 2/5 | django-11001, pytest-11143 | 29.0 | $3.89 |

**RAPS solved the exact same 2 tasks** the single agent did, and fixed none of the 3 it failed.
RAPS's lower total cost is **only** because RAPS caps at 30 rounds, clipping the baseline's runaway
sympy (71 rounds / $3.05); per-task RAPS is often *more* expensive (pytest 28 vs 17 rounds, sklearn
27 vs 19). The ~67 extra coordination LLM-calls/task buy nothing here.

### Why (the diagnosis, `runs/passk.py`)

pass@5 with **oracle gold files** (the exact files to edit handed to the agent), judged by the real
`FAIL_TO_PASS`, on the 6-task dev set = **1/6**. Only `psf__requests-1963` is winnable by
gpt-4o-mini even when told exactly which file to edit. The other 5 are a genuine **repair ceiling**.
Coordination cannot amplify a single-step repair the backbone cannot perform. This mirrors the
project's QA finding (fan-out ≈ 0 over single-agent; only objective verification helped).

---

## 1. How RAPS maps onto an agent loop (the core idea preserved)

The single hard invariant is **RAPS dynamic ad-hoc networking**; everything else is for performance.

`mini-swe-agent` is the **outer shell** (Docker env, bash/edit tools, history, submit protocol). RAPS
replaces only the **per-turn decision** — i.e. `DefaultAgent.query()`, the "one LLM → one action"
step. Each turn is one ad-hoc network round:

```
broker emits next-step NEED  (short imperative, from issue + recent command outputs)
   → text-embedding-3-small cosine-matches NEED vs each persona's subscription   (routing)
   → selected persona reactively specializes its guidance to the concrete issue  (reactive subscription)
   → persona publishes ONE bash action                                            (act)
```

There is **one shared workspace and one accumulating patch** — so there is no "vote over candidate
patches" problem. (That problem only arises for independent-trajectory methods like
self-consistency / best-of-N; see §6.)

Team = 5 personas in `RAPS/swe/team.py`: **Localizer, Reproducer, Editor, Verifier, Reviewer**.
Subscriptions use distinct action-oriented vocabulary so the embedding match is unambiguous
(verified: locate→Localizer, edit→Editor, test→Verifier, …).

---

## 2. Reproduce

```bash
# env: conda env `raps_swe` (py3.11) — swebench 4.1.0, datasets, docker SDK, openai, tiktoken.
# Backbone via ChatAnywhere (OpenAI-compatible). Key lives ONLY in:
#   RAPS/llm/chatanywhere_key.txt  AND  ~/.config/mini-swe-agent/.env (OPENAI_API_KEY / OPENAI_API_BASE)
# The global .env is what litellm/mini-swe read; it loads on `import minisweagent`.
# Docker on a SHARED 20-user box: always wrap docker commands in `sg docker -c '...'`,
# cap --max_workers, and NEVER global-prune (no `docker system prune -a`).

PY=/home/lirui/anaconda3/envs/raps_swe/bin/python
export HF_ENDPOINT=https://hf-mirror.com LITELLM_LOG=ERROR

# ---- single-agent baseline (gpt-4o-mini) ----
sg docker -c "$PY runs/run_baseline_metrics.py runs/eval20.txt runs/baseline_v2 4"

# ---- single-agent baseline (gpt-4o) ----
sg docker -c "$PY runs/run_baseline_gpt4o.py runs/eval20.txt runs/baseline_gpt4o 4"

# ---- RAPS (gpt-4o-mini, full 20) ----   args: <ids> <out> <workers> <step_limit>
sg docker -c "$PY RAPS/swe/run_swebench.py runs/eval20.txt runs/raps_full20 4 30"

# ---- RAPS (gpt-4o, 5-task slice) ----
sg docker -c "$PY RAPS/swe/run_swebench_gpt4o.py runs/raps5.txt runs/raps5_gpt4o 4 30"

# ---- resolved% (NOT in metrics.json — run the harness on preds.json) ----
IDS=$(paste -sd' ' runs/eval20.txt)
sg docker -c "$PY -m swebench.harness.run_evaluation \
  -d princeton-nlp/SWE-bench_Lite -s test -p runs/<dir>/preds.json -i $IDS \
  -id <run_id> --max_workers 4 --cache_level instance"
# read resolved_instances / resolved_ids from the emitted  <model>.<run_id>.json
```

Each runner writes per-task `metrics.json` (`{summary, rows}`) + a swebench-format `preds.json` +
per-task `<iid>/<iid>.traj.json` trajectories.

---

## 3. File map (SWE work only)

| Path | What |
|---|---|
| `runs/eval20.txt` | **fixed** 20-task stratified eval set (the only headline set) |
| `runs/dev6.txt` | 6 single-agent failures used as a dev set for error analysis |
| `runs/raps5.txt` | 5-task gpt-4o head-to-head slice |
| `runs/run_baseline_metrics.py` | single-agent baseline runner (gpt-4o-mini) + metrics |
| `runs/run_baseline_gpt4o.py` | single-agent baseline runner (gpt-4o) |
| `RAPS/swe/run_swebench.py` | RAPS runner (gpt-4o-mini) |
| `RAPS/swe/run_swebench_gpt4o.py` | RAPS runner (gpt-4o) — copy with `model_name=gpt-4o` |
| `RAPS/swe/agent.py` | `SWERAPSAgent(DefaultAgent)` — `query()` = one ad-hoc-network round; repro gate; apply_edit injection |
| `RAPS/swe/team.py` | the 5 personas (subscription + guidance) |
| `RAPS/swe/prompts.py` | centralized English prompts (iterate here): `SWE_RULES, BROKER_SYSTEM, REFINE_SYSTEM, …` |
| `runs/passk.py` | pass@k oracle-files diagnostic (the model-ceiling probe) |
| `runs/baseline_report.md` | written baseline analysis |
| `make_reviews_excel.py` → `RAPS_rebuttal_experiments.xlsx` | rebuttal experiment tables (from `reviews.md`) |
| `runs/baseline_v2/`, `runs/baseline_gpt4o/`, `runs/raps_full20/`, `runs/raps5_gpt4o/` | run outputs (metrics + preds + trajectories) |
| `<model>.<run_id>.json` (repo root) | swebench harness eval reports (resolved counts) |

---

## 4. Gotchas (will bite the next person)

1. **Format mode** — default `swebench.yaml` uses tool-calling, built for Claude; gpt-4o-mini returns
   bare text → `RepeatedFormatError` → empty patch. Both baseline runners and the RAPS runners use
   `swebench_backticks.yaml` + `model_class=litellm_textbased`. Keep that.
2. **`pull_timeout`** — default 120s is too short for the ~2.5 GB instance images; set to `1800`
   (already in every runner). When you pass `-c`, you must re-include the base config file.
3. **Resolved% is not in `metrics.json`** — the runner records rounds/tokens/cost/non-empty only.
   Always run the swebench harness on `preds.json` to get resolved (see §2).
4. **gpt-4o cost is mis-reported by the RAPS runner.** `RAPS/swe/agent.py` line ~29 hardcodes
   `PRICE_IN/PRICE_OUT` at **gpt-4o-mini** rates and computes `raps_stats.usd` there, so the
   `run_swebench_gpt4o.py` price edit is ignored. The **token counts are real** — recompute gpt-4o
   cost from `raps_stats.prompt_tokens` / `completion_tokens` (saved in every `traj.json`) at
   $2.50 / $10 per 1M. (This is how $3.89 was obtained, not the $0.23 the runner printed.)
5. **ChatAnywhere balance reserve.** The gateway reserves balance per request against its worst-case
   cost, so a tiny 5-token probe can pass while a real ~100k-token gpt-4o request is rejected with
   "账户余额过低不足以支持本次请求". Always probe with a **realistic-size** request before an expensive
   run, and keep ample headroom (a gpt-4o 20-task baseline ≈ $21).
6. **Shared machine.** `sg docker -c '...'`; cap `--max_workers`; only manage your own
   `sweb.eval.*` / `minisweagent-*` containers; never global-prune. Never put `pkill/pgrep -f
   <scriptname>` in a command whose own text contains that script name (self-kill, exit 144) — kill
   by PID instead.

---

## 5. State of the RAPS scaffold (what ships)

`multi_candidate` is **deferred / off** (`self.multi_candidate=False` in `agent.py`): its
revert-to-base-each-fire logic discarded the applied fix → empty patches, and pass@k proved the
self-repro selection signal is too weak. The **robust config that ships** = single `apply_edit`
SEARCH/REPLACE edits (accumulate, no revert; `ast`-checked) + reproduction gate + force-Reproducer
when none exists + repro-validation + issue-example repro + read-existing-tests. This reliably
produces non-empty patches. The mechanics are all fixed; the wall is the backbone, not the scaffold.

---

## 6. Open threads / suggested next steps (value-ordered)

1. **Scale RAPS-gpt-4o to all 20** (~$15–20) to firm up the negative result — 5 tasks is too small
   to be statistically conclusive, though the direction matches dev6 and full20.
2. **Task formed for coordination.** RAPS's value in the paper is shown on QA / multi-round reasoning;
   SWE's "single precise repair" step is exactly the regime where coordination doesn't help. A fair
   positive showcase likely needs a task shape that rewards intent-routing / role specialization.
3. **The one SWE lever: a real selection signal.** best-of-N independent trajectories selected by
   **running the repo's existing tests + an asserting issue-repro** (not LLM-judging patches). This
   is the only mechanism likely to move resolved on SWE, but it is real engineering and the repro
   selection signal was already shown to be weak under a weak backbone.

### Migrating the paper's baselines to SWE (design note)

Wrap every baseline the **same way RAPS is** — `mini-swe-agent` outer shell, the method swapped into
the per-turn decision — so the comparison is "same node, different coordination", not "different
harness". Under that wrapping: LLM-Debate / AutoAgents / Puppeteer / MAS-Zero (single evolving
solution / routing) migrate directly; SC / LLM-Blender / the MacNet graphs / GPTSwarm / AgentPrune /
G-Designer rely on **answer aggregation (voting)** which patches can't do → need test-based patch
selection (record as inspired-variants); AFlow / MaAS are high-cost (offline workflow search ×
Docker eval / supernet training data). See `RAPS_rebuttal_experiments.xlsx`.

---

## 7. Provenance

Backbone = ChatAnywhere OpenAI-compatible gateway (`gpt-4o-mini` / `gpt-4o`, embeddings
`text-embedding-3-small`, dim 1536). Dataset = `princeton-nlp/SWE-bench_Lite` (300 instances,
cached). Eval = official `swebench` Docker harness. See `swe-bench-raps-env` notes for the verified
environment. All numbers in this doc are from the eval reports committed at the repo root
(`gpt-4o-mini.baseline_v2.json`, `gpt-4o.baseline_gpt4o.json`, `gpt-4o-mini.raps_full20.json`,
`gpt-4o.raps5_gpt4o.json`).

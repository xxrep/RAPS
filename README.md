## Towards Adaptive, Scalable, and Robust Coordination of LLM Agents:<br>A Dynamic Ad-Hoc Networking Perspective

<img src="./image/raps.png" width="100%" alt="RAPS Overview"/>

[[📄]](https://arxiv.org/abs/2502.xxxxx) _Towards Adaptive, Scalable, and Robust Coordination of LLM Agents: A Dynamic Ad-Hoc Networking Perspective_

Multi-agent architectures built on large language models (LLMs) have demonstrated the potential to realize swarm intelligence through well-crafted collaboration. However, the substantial burden of manual orchestration inherently raises an imperative to automate the design of agentic workflows. We frame such an agent coordination challenge as a classic problem in dynamic ad-hoc networking: _How to establish adaptive and reliable communication among a scalable number of agentic hosts?_ In response to this unresolved dilemma, we introduce **RAPS**, **a Reputation-Aware Publish-Subscribe paradigm for adaptive, scalable, and robust coordination of LLM agents**. RAPS grounds its ad-hoc coordination fabric in a Distributed Content-Centric Protocol, allowing agents to exchange messages based on their declared intents rather than predefined communication topologies. Beyond such a flexible substrate, RAPS further incorporates two critical overlay mechanisms: (i) Reactive Subscription, which enables agents to refine their intents on the fly; and (ii) Bayesian Reputation, which empowers agents with a local watchdog to isolate malicious peers. Extensive experiments on five benchmarks showcase that RAPS effectively reconciles adaptivity, scalability, and robustness within a unified coordination framework.

---

## 🧭 How the implementation maps to the paper

The whole coordination loop lives in **`RAPS/core/coordinator.py`** (`RAPSCoordinator`). One task is solved by repeating these per-step phases over a team of agent "hosts":

| Paper mechanism | Where it lives |
|---|---|
| **Distributed Content-Centric Protocol** (route by declared intent, not fixed topology) | `Node.publish()` → `Node.broker_route()` (embedding match of the predicted next-intent to every agent's subscription) |
| **Reactive Subscription** (agents refine intent on the fly) | `Node.refine_system_prompt()`, called each step |
| **Bayesian Reputation + local watchdog** (isolate malicious peers) | `Node.watchdog_evaluate()` + `RAPS/graph/reputation.py` (`ReputationManager`, CONFIDANT-style) + the routing **trust gate** |
| **Adaptive team** (membership / recruitment / capacity adapt to the task) | `RAPSCoordinator` dynamic membership, `RAPS/agents/seed_pool.py` recruitment, capacity-adaptive fan-out |

Per-step flow inside `RAPSCoordinator.run()`:
`ingest inboxes → reactive subscription → publish → watchdog + first-hand reputation → (optional) gossip + trust gate → consensus / capacity check → broker routing (+ recruitment)`, then a **Final Answerer** aggregates all publications.

### Adaptive agent team
- **Dynamic membership** — only agents the broker routes to become active next step; the rest stay dormant. Team activity per step is recorded in each task log under `team.active_per_step`.
- **On-demand recruitment** — when a seed specialist (declared in `RAPS/agents/seed_pool.py`) matches the predicted intent better than any current teammate, it is instantiated and joins the team (`--dynamic_recruit`, capped by `--max_team_size`, reset per task).
- **Adaptive capacity** — broker fan-out (`top_k`) scales with task difficulty (`--adaptive_capacity`, bounded by `--max_top_k`).
- **Adaptive termination** — the round stops early when active agents reach a **consensus** answer (a real signal, replacing the previous dead heuristic).

### Robustness (Bayesian reputation)
The local watchdog vets every incoming message and feeds a per-agent Bayesian reputation (`ReputationManager`). With `--reputation_gate`, an agent **drops messages from peers it distrusts**, isolating malicious hosts; with `--second_hand_gossip`, agents exchange first-hand reports so a peer can be isolated even before a direct bad interaction. See `experiments/run_robustness.py`.

---

## 🛠️ Setup

```bash
git clone <this-repo> RAPS && cd RAPS

# 1) App environment (Python 3.12)
conda create -y -n raps python=3.12
conda run -n raps pip install -r requirements.txt

# 2) LLM gateway — gpt-4o-mini via the Azure/modelhub gateway.
#    Provide the key out-of-repo (env var or a gitignored file):
export AZURE_OPENAI_API_KEY="<key>"
# optional overrides (defaults shown):
#   AZURE_OPENAI_ENDPOINT=https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl
#   AZURE_OPENAI_API_VERSION=2024-02-01
# To use the public OpenAI endpoint instead: export RAPS_LLM_BACKEND=GPTChat

# 3) Embedding service (Qwen3-Embedding-4B on GPU 0, OpenAI-compatible) — see serving/README.md
bash serving/start_embedding.sh &      # serves http://127.0.0.1:8200/v1
```

The broker's semantic routing uses the local **Qwen3-Embedding-4B** service rather than a cloud embedding API. See [`serving/README.md`](serving/README.md) for details (it runs in its own `embed` conda env with a CUDA-12-compatible torch).

## 📚 Datasets
Local copies live under `raps_data/` (`gsm8k/`, `humaneval/`, `MMLU/`). GSM8K and HumanEval ship as JSONL; MMLU data can be fetched with `raps_data/MMLU/download.py`.

> Note: the data package was renamed `datasets/ → raps_data/` so it no longer shadows the pip `datasets` library.

## 🚀 Run

```bash
# GSM8K (base RAPS)
python experiments/run_gsm8k.py --llm_name gpt-4o-mini-2024-07-18 --max_steps 3 --limit 50

# Adaptive team (recruit specialists + capacity-adaptive fan-out)
python experiments/run_gsm8k.py --dynamic_recruit --adaptive_capacity --limit 50

# Robustness defenses on
python experiments/run_gsm8k.py --reputation_gate --second_hand_gossip --limit 50

# MMLU / HumanEval
python experiments/run_mmlu.py --limit 159
python experiments/run_humaneval.py --limit 20
```

Common flags: `--max_steps`, `--top_k`, `--sim_threshold`, `--entry_index`, `--start/--limit`,
`--reputation_gate`, `--second_hand_gossip`, `--dynamic_recruit`, `--adaptive_capacity`,
`--max_team_size`, `--max_top_k`.

### Robustness evaluation
```bash
python experiments/run_robustness.py --limit 12 --num_malicious 3 --top_k 2 --max_steps 3
```
Injects disguised malicious peers (whose subscriptions look helpful but whose outputs are
deliberately wrong) and compares `no_defense`, `watchdog`, and `watchdog+reputation`. Results
are saved to `result/robustness/`.

Illustrative run (GSM8K, gpt-4o-mini, 12 questions, 3 malicious peers, top_k=2):

| condition | accuracy | malicious msgs blocked | avg. malicious reputation |
|---|---|---|---|
| no_defense | 0.917 | 0 | – |
| watchdog | 0.833 | 6 | 0.667 |
| **watchdog + reputation** | **0.917** | **10** | **0.472** |

The defense's effect is clearest in **how aggressively malicious peers are isolated** (messages
blocked rises 0→6→10; the full defense also drives the malicious peers' reputation lowest). On a
small, easy slice the Final Answerer already absorbs a minority of bad messages, so raw accuracy
sits near the no-malicious ceiling; the isolation metrics make the reputation mechanism's value
explicit. Scale `--limit` / `--num_malicious` for sharper accuracy separation.

## 📊 Output
- **Logs**: per-task traces (prompts, publications, broker decisions, per-step reputation snapshots, team summary) in `logs/`.
- **Results**: metrics in `result/<benchmark>/`; cost/token totals printed to console.

## 🗂️ Layout
```
RAPS/core/coordinator.py     # the RAPS pub/sub coordination loop (RAPSConfig + RAPSCoordinator)
RAPS/graph/node.py           # Node: publish / refine_system_prompt / watchdog / broker_route / check_trust
RAPS/graph/reputation.py     # ReputationManager (CONFIDANT-style Bayesian reputation)
RAPS/agents/                 # MathAgent, CodeWriting, AnalyzeAgent, AdversarialAgent, seed_pool
RAPS/llm/azure_chat.py       # Azure gateway chat + local embedding-service client
serving/                     # Qwen3-Embedding-4B OpenAI-compatible service
experiments/                 # run_gsm8k / run_mmlu / run_humaneval / run_robustness
RAPS/legacy/                 # forked GCN-graph code, kept for reference, not used
```

## 📜 Acknowledgement
This project builds on insights from dynamic network theory, multi-agent systems, and large language model research. We thank the authors of GSM8K, MMLU, and HumanEval for their benchmarks.

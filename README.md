## Towards Adaptive, Scalable, and Robust Coordination of LLM Agents:<br>A Dynamic Ad-Hoc Networking Perspective

<img src="./image/raps.png" width="100%" alt="RAPS Overview"/>

[[📄]](https://arxiv.org/abs/2602.08009) _Towards Adaptive, Scalable, and Robust Coordination of LLM Agents: A Dynamic Ad-Hoc Networking Perspective_

**RAPS** is a Reputation-Aware Publish-Subscribe paradigm for coordinating LLM agents. Agents exchange
messages by declared intent rather than through a predefined topology (a Distributed Content-Centric
Protocol), refine those intents as a task unfolds (Reactive Subscription), and keep a local watchdog and
Bayesian reputation that isolates unreliable peers (Bayesian Reputation).

The coordination loop is `RAPS/core/coordinator.py`; the protocol parameters behind every reported run
are declared once in `RAPS/config.py`.

---

## Install

```bash
conda create -y -n raps python=3.12
conda activate raps
pip install -r requirements.txt
```

## Configure the backbone

Keys and endpoints come from the environment. Nothing is hardcoded, and no key belongs in the
repository — a key may also be placed in a gitignored `RAPS/llm/*_key.txt`.

**Required.** The backbone every agent calls:

```bash
export AZURE_OPENAI_API_KEY="<key>"
export AZURE_OPENAI_ENDPOINT="<your Azure-compatible endpoint>"
```

To call an OpenAI-compatible endpoint instead:

```bash
export RAPS_LLM_BACKEND=GPTChat
export OPENAI_API_KEY="<key>"
export OPENAI_BASE_URL="https://api.openai.com/v1"   # only if not the official API
```

**Optional**

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude backbone for the cross-backbone comparison |
| `QWEN_BASE_URL` | Qwen3-32B served locally behind an OpenAI-compatible server |
| `AZURE_OPENAI_API_VERSION` | API version, default `2024-02-01` |
| `RAPS_TRACE_HEADER` | `"Name: value"` request header, if your gateway requires one |
| `RAPS_TEMPERATURE` | override the shared decoding temperature |
| `RAPS_PRICE_IN` / `RAPS_PRICE_OUT` | price a run at other USD-per-1M-token rates |

## Configure the tools

The harness grants three tools. Each is reached through the environment as well, and each degrades
into plain reasoning if its service is absent, so a run never fails for want of one.

### Broker embeddings — always used

Routing matches a forwarding query against the agents' subscriptions, so one embedding model is needed.
Naming a `text-embedding-*` model sends the call to the OpenAI API; any other name goes to an
OpenAI-compatible service you host:

```bash
# hosted
export EMBED_MODEL=text-embedding-3-small
export EMBED_API_KEY="<key>"                 # or reuse OPENAI_API_KEY

# local (Qwen3-Embedding-4B on GPU 0, see serving/README.md)
export EMBED_MODEL=qwen3-embedding-4b
export EMBED_BASE_URL=http://127.0.0.1:8200/v1
export EMBED_MODEL_PATH=Qwen/Qwen3-Embedding-4B    # local path or hub id
export EMBED_PORT=8200                             # EMBED_HOST=0.0.0.0 to serve other machines
bash serving/start_embedding.sh &
```

### Python executor — code generation and program-aided arithmetic

In-process execution with a per-call timeout; no service to start. It scores HumanEval, repairs code
against the public doctests, and computes the arithmetic answer by running Python instead of generating
it:

```bash
# execute the candidate against the public doctests and repair it from real failures
python experiments/run_humaneval.py --code_verify --code_verify_max_iters 3 --edge_cases

# program-aided arithmetic on the mathematical benchmarks
python experiments/run_math.py --dataset gsm8k --tool_verify
```

`--edge_cases` adds model-written edge tests as a secondary signal; the public doctests always
dominate, so a written test can never override the specification.

### Retrieval — the knowledge roles on MMLU

The knowledge roles retrieve from a local Wikipedia passage index over Elasticsearch. Start
Elasticsearch, put the dump in place, and build the index once:

```bash
export ES_URL=http://127.0.0.1:9200          # the Elasticsearch host
export WIKI_DIR=wiki/20231101.en             # the parquet dump
export WIKI_INDEX=wiki_kstem                 # the index the retriever queries

python RAPS/tools/search/index_wiki.py       # ~128-word passages, then the kstem re-index
```

Verify it answers, and run MMLU with retrieval off for the comparison:

```bash
python -c "from RAPS.tools.search.bm25_retriever import bm25_retrieve; \
print(bm25_retrieve(['injured State responsibility breach obligation'], k=5))"

python experiments/run_mmlu.py --no_react    # same pool, no retrieval
```

## Data

`raps_data/` ships GSM8K and HumanEval. Fetch MMLU with `raps_data/MMLU/download.py`; drop
`SVAMP.json` from [arkilpatel/SVAMP](https://github.com/arkilpatel/SVAMP) into `raps_data/svamp/`, and
`dev.json` from [google-deepmind/AQuA](https://github.com/google-deepmind/AQuA) into `raps_data/aqua/`.

## Run the main experiments

Every runner takes its protocol parameters from `RAPS/config.py` — one parameter set, used unchanged on
all five benchmarks — so an ablation is an explicit override rather than a different default.

```bash
# The five benchmarks
python experiments/run_math.py --dataset gsm8k --limit 1319
python experiments/run_math.py --dataset svamp --limit 1000
python experiments/run_math.py --dataset aqua  --limit 254
python experiments/run_mmlu.py --limit 153
python experiments/run_humaneval.py --limit 164

# Cross-backbone: one backbone per run, or mixed across the pool
python experiments/run_math.py --dataset gsm8k --llm_name claude-sonnet-5
python experiments/run_math.py --dataset gsm8k --llm_name Qwen/Qwen3-32B
python experiments/run_math.py --dataset gsm8k \
    --role_model "Math Solver=claude-sonnet-5" --role_model "Inspector=Qwen/Qwen3-32B"

# Budget-capped coordination: a per-task ceiling on prompt + completion tokens
python experiments/run_math.py --dataset gsm8k --budget_tokens 20000

# Robustness on MMLU: five adversary types under three defence conditions
python experiments/run_robustness.py --adversary overt    --truthful 3 --num_adversary 2
python experiments/run_robustness.py --adversary covert   --collusion both
python experiments/run_robustness.py --adversary adaptive --degrade_p 0.3
python experiments/run_robustness.py --dataset gsm8k --adversary sleeper

# Open membership: agents join and depart mid-episode
python experiments/run_membership.py --mode churn --leave_prob 0.25
python experiments/run_membership.py --mode targeted
python experiments/run_membership.py --mode newcomer --arrive_at 2

# Mechanism ablations, and the naive pool of five generic profiles
python experiments/run_math.py --dataset gsm8k --no_reputation_gate
python experiments/run_math.py --dataset gsm8k --no_watchdog
python experiments/run_math.py --dataset gsm8k --naive_pool

# Repository-level repair: the mean over four independent repeats
python RAPS/swe/run_swebench.py --dataset lite --out swe_runs/raps_4o_lite \
    --model gpt-4o --workers 4 --repeats 4
```

Common flags: `--llm_name`, `--start/--limit`, `--max_steps`, `--top_k`, `--sim_threshold`,
`--entry_index`, `--budget_tokens`, `--dynamic_recruit`, `--adaptive_capacity`. Each runner's `--help`
lists the rest.

## Output

Per-task traces go to `logs/` (publications, subscription refinements, broker decisions, reputation
snapshots), metrics to `result/<benchmark>/`, and token and cost totals to the console. The same traces
feed the diagnostics:

```bash
python analysis/cost_decomposition.py "logs/log_gsm8k_*.json"
python analysis/routing_diagnostics.py "logs/log_mmlu_*.json"
python analysis/watchdog_eval.py "logs/log_gsm8k_*.json" --domain gsm8k
```

## Layout

```
RAPS/config.py            protocol parameters, agent pools, ablation flags
RAPS/core/coordinator.py  the publish-subscribe coordination loop
RAPS/graph/node.py        publish, subscription refinement, watchdog, broker routing
RAPS/graph/reputation.py  Beta reputation, the witness scheme, the routing gate
RAPS/agents/              benchmark agents, the adversary types, the recruitable seeds
RAPS/llm/                 backbone and embedding backends
RAPS/tools/               python executor, Wikipedia retrieval
RAPS/swe/                 repository-level repair
experiments/              the benchmark runners
analysis/                 diagnostics derived from the task traces
serving/                  local embedding service
```

## Acknowledgement

This project builds on insights from dynamic network theory, multi-agent systems, and large language
model research. We thank the authors of GSM8K, MMLU, HumanEval, SVAMP, AQuA and SWE-bench for their
benchmarks.

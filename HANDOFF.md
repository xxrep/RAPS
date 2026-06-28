# RAPS — Handoff & Design Document

**RAPS = Reputation‑Aware Publish‑Subscribe** coordination of LLM agents.
Paper: *“Towards Adaptive, Scalable, and Robust Coordination of LLM Agents: A Dynamic Ad‑Hoc Networking Perspective.”*

This document is the single source of truth for: (1) the full framework design, (2) the exact
agent pool + prompts for each benchmark, (3) one fully‑traced concrete case per benchmark
(subscription → publication → broker routing → final answer), (4) the honest performance state,
and (5) how to run everything.

> TL;DR of where the project stands:
> - **GSM8K ≈ 0.922**, **HumanEval ≈ 0.887** (the one clear engineered win), **MMLU ≈ 0.76–0.84** (sample‑dependent).
> - The multi‑agent coordination, a working **ReAct Wikipedia retrieval** path, and a local
>   **27.2 M‑passage Elasticsearch/BM25 index** are all implemented and verified.
> - **Honest finding:** on MMLU, *no* design lever we tried (multi‑agent fan‑out, ReAct retrieval,
>   CoT depth, arbitration routing) moves net accuracy — they each net ≈ 0. The remaining MMLU
>   failures are capability‑bound (the model reasons confidently down a wrong frame, esp. law).
>   Retrieval genuinely helps *fact/definition* questions but breaks an equal number elsewhere.

---

## 0. Environment & Infrastructure

| Piece | Detail |
|---|---|
| Working dir | `/opt/tiger/RAPS/RAPS` (repo); experiments in `/opt/tiger/RAPS/experiments` |
| Python env | conda env **`raps`** (Python 3.12). Embedding service uses a separate env **`embed`** (torch 2.4.1+cu121, driver 12.2). |
| LLM backbone | `gpt-4o-mini-2024-07-18` via the internal Azure/modelhub gateway. Wrapper: `RAPS/llm/azure_chat.py` (`AzureGPTChat`). Endpoint `https://aidp-i18ntt-sg.byteintl.net/...` (office net: `...tiktok-row.net`). Key in `RAPS/llm/azure_key.txt` (gitignored). |
| Embeddings | **Qwen3‑Embedding‑4B** served locally on **GPU 0**, FastAPI OpenAI‑compatible, port **8200**, dim **2560**, bf16. Code: `serving/embedding_server.py`, launch `serving/start_embedding.sh`. Used for broker routing. |
| Retrieval | **Elasticsearch 8.15** (bundled JDK 22, single‑node, security off). Index **`wiki_kstem`** = 27,185,091 Wikipedia passages (~128 words each, 2023‑11 English dump). BM25. Code: `RAPS/tools/search/bm25_retriever.py`. |
| Proxy | **All downloads / ES / embedding calls must set `no_proxy='*'`** (the gateway is reached *with* proxy; localhost services *without*). |
| Determinism | `RAPS_TEMPERATURE=0` forces temp 0, but the gateway is **not** perfectly deterministic — small‑sample accuracy wobbles ±1–2 questions. Always compare on the **same** question set. |

Quick start:
```bash
conda activate raps
export RAPS_TEMPERATURE=0 no_proxy='*' NO_PROXY='*'
# embedding service (GPU0) and Elasticsearch must already be running
python experiments/run_mmlu.py     --limit 100 --top_k 3 --sim_threshold 0.40 --max_steps 5
python experiments/run_gsm8k.py    --limit 200
python experiments/run_humaneval.py --limit 160
```

---

## 1. Framework Architecture

RAPS models a team of LLM agents as a **content‑centric publish/subscribe network**. There is no
fixed call graph; who speaks next is decided at runtime by *semantic routing* over what was just
said. Three pillars (from the paper), all implemented:

1. **Distributed Content‑Centric Protocol** — agents `publish()` messages; a **broker** routes each
   publication to the most relevant downstream agents by embedding similarity (`broker_route()`).
2. **Reactive Subscription** — before acting, each active agent **re‑specializes its own persona**
   to the current question/context (`refine_system_prompt()`).
3. **Bayesian Reputation** — a CONFIDANT‑style **watchdog** validates each publication; a
   **reputation gate** can block low‑reputation agents; **second‑hand gossip** propagates trust.

### 1.1 Core objects

| Object | File | Role |
|---|---|---|
| `RAPSCoordinator`, `RAPSConfig` | `RAPS/core/coordinator.py` | Orchestrates the layered execution loop + final decision. |
| `Node` (base agent) | `RAPS/graph/node.py` | `publish()`, `broker_route()`, `_match_by_embedding()`, `refine_system_prompt()`, watchdog + reputation. |
| `AnalyzeAgent` | `RAPS/agents/analyze_agent.py` | MMLU agent. **Now also hosts the ReAct retrieval loop** (`_react_gen`). |
| `MathAgent` | `RAPS/agents/math_solver.py` | GSM8K agent. |
| `CodeWriting` | `RAPS/agents/code_writing.py` | HumanEval agent. |
| `RetrieverAgent` | `RAPS/agents/retriever_agent.py` | Optional dedicated retrieval host (not wired into MMLU by default; superseded by per‑agent ReAct). |
| Prompt sets | `RAPS/prompt/{mmlu,gsm8k,humaneval}_prompt_set.py` | Roles, role descriptions, decision/constraint prompts. |

### 1.2 The execution loop (`RAPSCoordinator.run`)

```
run(task):
  _top_k = _estimate_top_k(task)              # = cfg.top_k unless adaptive_capacity
  active_agents = [entry_agent]               # STEP 1 always starts with ONE entry host (entry_index)
  for step in 1..max_steps:
     _ingest_inboxes(active_agents)           # pull messages routed to me last step into my history
     _reactive_subscription(active_agents)    # each active agent refines its persona to the question
     pubs = _publish(active_agents)           # each active agent produces ONE message (ReAct loop if enabled)
     [watchdog] validate pubs, update reputation, optionally block          (if use_watchdog)
     [consensus] if all active agents agree -> stop early
     next_active = _route(pubs)               # broker fans each publisher out to its top_k matches
     active_agents = next_active
  final_output = _final_decision(task, all_publications)
```

Key behaviours:

- **`top_k` is the broker fan‑out** (how many downstream agents each publisher routes to). With
  `top_k=1` the whole thing degenerates to a **serial chain** (`active_per_step=[1,1,1,1,1]`).
  With `top_k=3` it becomes a genuine **fan‑out** committee (e.g. `[1,3,4,4,4]`), and heterogeneous
  roles (incl. the Critic) actually participate. **This was a real bug we found:** the original
  MMLU baseline had been run at `top_k=1` (degenerate chain); the configured design is `top_k=3`.
- **Step 1 is always a single entry agent** (`entry_index`, default 0 = first role in the roster);
  fan‑out begins at step 2.

### 1.3 Broker routing — `broker_route()` + `_match_by_embedding()` (`RAPS/graph/node.py`)

Routing is a **two‑stage** process (this is the heart of “content‑centric”):

**Stage 1 — predict the needed role (1 LLM call).** The broker looks at the publications so far and
writes a short natural‑language description of *the downstream role best suited to continue*
(“semantic router”). In `broker_mode="gap"` it instead names the *missing capability* (e.g. “needs
independent verification / code execution”).

**Stage 2 — embedding match (`_match_by_embedding`).** That predicted‑role text is embedded
(Qwen3) and compared by **cosine similarity** to every candidate agent’s **subscription string**:

```
subscription = "Your Role: {role}.  Your Capabilities: {capabilities}  Your Interests: {interests}"
scored = sorted([(agent, cosine(predicted_vec, agent_sub_vec)) ...], desc)
picked = [a for a,s in scored[:top_k] if s >= sim_threshold]
if not picked: picked = [scored[0][0]]      # FALLBACK: always engage the single best match
```

Two subtleties we learned the hard way:
- **The fallback to the single best match** is what turns a high threshold into a chain: if no
  candidate clears `sim_threshold`, only one agent is engaged → `[1,1,1,1,1]`.
- **Similarity scores are bimodal.** For “specialist” questions only one role scores ~0.9 and the
  rest crater to ~0.3, so at `sim_threshold=0.40` many questions still collapse to a chain. Lowering
  the threshold *does* force fan‑out but pulls in **irrelevant** roles and **hurts** accuracy
  (sweep: 0.40→0.74, 0.30→0.67, 0.20→0.66). **0.40 is the sweet spot; do not lower it.**

### 1.4 Reactive subscription — `refine_system_prompt()`

Before publishing, each active agent makes **one LLM call** that rewrites its own system prompt to
be hyper‑specific to the current question (e.g. a generic *Knowledge Expert* becomes “a legal expert
in international state‑responsibility law…”). The refined persona is what then answers.

### 1.5 Watchdog + Bayesian reputation (optional, off by default in the benchmarks)

- `watchdog_evaluate()` — an upstream agent judges whether a downstream publication is valid given
  the question + existing info; updates a Beta(α,β) reputation per (observer, target).
- `reputation_gate` — blocks publications from agents whose trust `α/(α+β)` is below threshold.
- `second_hand_gossip` — agents merge each other’s reputation estimates (with a discount).
- These power the **robustness/adversarial** experiments (a disguised malicious `AdversarialAgent`
  can be injected and the reputation system learns to suppress it).

### 1.6 Final decision — `_final_decision()`

1. Concatenate **all** publications across all steps into a `Process History`.
2. *(tool/code domains only)* run **tool verification**: `tool_verify` (GSM8K → PAL Python execution)
   or `code_verify` (HumanEval → run doctests + repair). The executed result is injected as an
   authoritative note.
3. The **Final Answerer** refines its persona, reads the task + full history, and emits the final
   answer (a single letter for MMLU; the number for GSM8K; the code block for HumanEval).
4. `answer_extractor` (per‑dataset `postprocess_answer`) parses the final string.

### 1.7 ReAct retrieval (new — MMLU knowledge roles)

Implemented in `AnalyzeAgent._react_gen` (`RAPS/agents/analyze_agent.py`) and enabled per‑agent in
`run_mmlu.initialize_agents_from_set`. **Each knowledge‑intensive agent can autonomously retrieve**
while reasoning (true ReAct — *the reasoning agent itself decides when and what to search*):

```
loop up to react_max_search (=2) times:
   response = llm(messages)
   if response contains a line "SEARCH: <keywords>":
       obs = bm25_retrieve([keywords], k=5)      # local Wikipedia, top-5 passages
       append assistant(response) + user("Observation: <obs>") to messages ; continue
   else: break
```

- **Enabled roles** (`KNOWLEDGE_ROLES`): Knowledge Expert, Historian, Doctor, Lawyer, Economist,
  Psychologist. Pure‑reasoning roles (Critic, Mathematician, Programmer) keep plain reasoning.
- **Query style:** the prompt instructs the agent to emit **discriminative keywords**, not a
  natural‑language sentence (this matters a lot for BM25 — see §5).
- **Verified effective at retrieval:** agents land on exactly the right Wikipedia articles
  (e.g. *Securities Act of 1933*, *Equitable conversion*, *Pyruvate dehydrogenase complex*).
- **But net accuracy impact = 0** (see §6): it fixes fact/definition questions and breaks an equal
  number where retrieved context misleads, and fires spuriously on questions that need no facts.

---

## 2. Per‑Benchmark Configuration

### 2.1 GSM8K — grade‑school math

| | |
|---|---|
| Agent class | `MathAgent` |
| Roster (`ROLES`) | **Math Solver, Mathematical Analyst, Programming Expert, Inspector** |
| Seed pool (recruitable) | Algebra Specialist, Arithmetic Verifier, Word Problem Translator, Unit & Constraint Checker |
| Config | `domain=gsm8k`, `max_steps≈3–5`, `top_k=3`, `sim_threshold≈0.70`, `entry_index=0` |
| Special | **PAL** Program‑Aided execution available; `tool_verify` runs Python at the final step |
| Few‑shot | per‑role exemplars (`FEW_SHOT_DATA`) |
| **Performance** | **≈ 0.922** (345/374). `tool_verify` net **+0** — the apparent gains were temperature noise, not the tool. |

Role prompts (`RAPS/prompt/gsm8k_prompt_set.py`):
- **Math Solver** — “You are a math expert… give your own step‑by‑step solving process based on hints. The last line contains only the final result without units, e.g. `The answer is 140`.”
- **Mathematical Analyst** — “…first analyze the problem step by step with variables as letters, then substitute values and compute. Last line: `The answer is 140`.”
- **Programming Expert** — “…integrate step‑by‑step reasoning and Python code; write a no‑arg function returning the result; last line assigns the return value to `answer`. Respond only with a Python code block.”
- **Inspector** — “…check whether the logic/calculation and any code are correct; give your own step‑by‑step solution. Last line: `The answer is 140`.”

### 2.2 MMLU — 4‑option multiple choice (57 subjects)

| | |
|---|---|
| Agent class | `AnalyzeAgent` (+ ReAct retrieval on knowledge roles) |
| Roster (`ROLES`) | **Knowledge Expert, Critic, Mathematician, Psychologist, Historian, Doctor, Lawyer, Economist, Programmer** + a dedicated **Final Answerer** |
| Seed pool | Statistician, Philosopher, Biologist, Computer Scientist |
| Config | `domain=mmlu`, `max_steps=5`, `top_k=3`, `sim_threshold=0.40`, `entry_index=0` |
| ReAct retrieval | ON for {Knowledge Expert, Historian, Doctor, Lawyer, Economist, Psychologist} |
| **Performance** | base model single‑solver ≈ 0.807; chain (top_k=1) ≈ 0.837; clean top_k=1 on 100 Qs = 0.760; **ReAct net +0 (76=76)**; fan‑out / CoT / arbitration all net ≈ 0. |

Role descriptions (`RAPS/prompt/mmlu_prompt_set.py`, abbreviated — all are 2–3 sentences):
- **Knowledge Expert** — interpret the question, identify the core concept, recall relevant facts, weigh options.
- **Critic** — review others’ reasoning, spot logical gaps/misreadings, propose a clearer line to the correct option.
- **Mathematician** — quantitative/logic reasoning; recall formulas/definitions; check calculations.
- **Psychologist** — psychology/sociology/philosophy theories and empirical findings.
- **Historian** — cultural/political/economic/social context and chronology.
- **Doctor** — clinical facts, mechanisms, guidelines; medical plausibility/safety.
- **Lawyer** — legal principles, definitions, precedents; legality/doctrinal consistency.
- **Economist** — micro/macro/finance models, incentives, empirical patterns.
- **Programmer** — CS/software/physics fundamentals; precise technical reasoning, edge cases.

Knowledge roles additionally receive the **ReAct retrieval instruction** (emit `SEARCH: <keywords>`
to verify a decisive fact via Wikipedia; keywords not sentences; ≤2 searches).

The **Final Answerer** persona: *“You are the top decision‑maker, good at analyzing and summarizing
others’ opinions, finding errors and giving final answers.”* — and is constrained to output **only a
single letter** A/B/C/D.

### 2.3 HumanEval — code generation

| | |
|---|---|
| Agent class | `CodeWriting` |
| Roster (`ROLES`) | **Project Manager, Algorithm Designer, Programming Expert, Test Analyst, Bug Fixer** (extra: Normal/Stupid Programmer for robustness tests) |
| Seed pool | Edge Case Analyst, Complexity Optimizer, Spec & Signature Reviewer |
| Config | `domain=humaneval`, `max_steps=5`, `top_k≈1–3`, `sim_threshold≈0.60` |
| Special | **`code_verify`** (execute public doctests → repair loop, ≤N iters) + **Edge‑Case Writer / Code Verifier** |
| **Performance** | baseline **0.781** (125/160) → **≈ 0.887** with `code_verify`+`edge_cases` (recovers 17/35 failures, **0 breakage**). **The one clear engineered win** — because code execution is an *objective* verifier. |

Role prompts (`RAPS/prompt/humaneval_prompt_set.py`):
- **Project Manager** — oversee overall structure; suggest a concise correct design (≤50 words).
- **Algorithm Designer** — specify algorithm/classes/functions; pseudocode if complex (≤50 words).
- **Programming Expert** — write the full implementation (restate the signature); Python code block only.
- **Test Analyst** — point out edge cases, boundary conditions, potential errors (≤50 words).
- **Bug Fixer** — produce improved full implementation based on the design + test feedback; code block only.

### 2.4 Where every prompt lives + the two framework‑level meta‑prompts

**Prompt locations at a glance:**

| Prompt | What it is | Where (file · symbol) |
|---|---|---|
| Role / capability prompts | each agent's base persona (the §2.1–2.3 descriptions) | `RAPS/prompt/{mmlu,gsm8k,humaneval}_prompt_set.py` · `ROLE_DESCRIPTION` |
| Answer/decision constraints | output‑format rules (e.g. “first line is one letter”) | same files · `get_constraint` / `get_decision_constraint` |
| ReAct retrieval instruction | the `SEARCH:` tool spec appended to knowledge roles | `experiments/run_mmlu.py` · `REACT_RETRIEVAL_INSTRUCTION` |
| **Reactive‑subscription (REFINE) meta‑prompt** | ① rewrites an agent's persona per question | `RAPS/graph/node.py` · `refine_system_prompt()` |
| **Broker (routing) meta‑prompt** | ④ predicts the needed next role | `RAPS/graph/node.py` · `broker_route()` |

The two framework‑level meta‑prompts are *not* per‑dataset — they drive ① and ④ in every trace.
Verbatim:

**① Reactive‑subscription / REFINE** (`refine_system_prompt`, default `solve` mode):
```
[system] You are an expert prompt refiner who tailors agent instructions to the current
         question and context based on the evolving workflow.
[user]   ### GOAL
         Your task is to dynamically specialize an LLM agent's persona to perfectly align with
         the specific needs of the current problem state. Do NOT simply summarize the old
         profile. Instead, evolve the agent's intent to be highly specific to the immediate context.

         ### INPUT DATA
         1. **Base Profile (Starting Point):** {agent.system_prompt}
         2. **Current Task/Question:** {question}
         3. **Interaction Context (Message Flow):** {context}

         ### INSTRUCTIONS
         - Analyze the Current Task and Interaction Context to identify what specific expertise,
           constraints, or output format is missing or needed right now.
         - Ignore generic traits in the Base Profile if they are not relevant to the current step.
         - Rewrite the System Instruction to explicitly guide the agent on *how* to process the
           specific input in the context.
         - The new instruction should act like a focused 'Mission Briefing' for this step.

         ### OUTPUT FORMAT
         Return ONLY the refined system instruction string. No explanations.
```
*(An alternate `persona` mode also exists — same idea but with HARD CONSTRAINTS forbidding the
refiner from solving the problem or stating any answer; used when you want lens‑only specialization.)*
The returned string becomes the agent's `refined_prompt`, which is the `system` for its ③ PUBLISH call.

**④ Broker / routing** (`broker_route`, default mode):
```
[system] You are a semantic router. You bridge the gap between current progress and required expertise.
[user]   You are coordinating a multi-agent system.
         Choose the most suitable downstream role based on the available role options.
         Previous agents' outputs:
         {concatenated publications so far}
         Available downstream roles (current agent excluded):
         {numbered list of "[agent_id] <subscription string>"}
         Write a concise role description grounded in the options above. Do not write a task plan.
         Keep it under 100 words.
```
The model's reply is the **predicted next role** (the ④ BROKER LLM output in §3). It is then embedded
and cosine‑matched (`_match_by_embedding`) against each candidate's subscription string to pick the
top‑`top_k` receivers ≥ `sim_threshold`.

There is also a **`gap` mode** that asks for the *missing* capability instead of the best next role —
it explicitly prefers an independent/exact verifier when the work so far is only natural‑language
reasoning (this is what biases GSM8K toward engaging the Inspector / a code verifier):
```
[user]   You are coordinating a multi-agent system.
         Identify the single most important capability that is STILL MISSING to finish the task,
         then describe the downstream role best suited to provide it.
         If the work so far is only natural-language reasoning with no independent/exact
         verification (e.g. no code execution), prefer a role that provides that verification.
         {previous outputs} … {available roles} … (concise, <100 words, no task plan)
```

---

## 3. Fully‑Traced Concrete Cases — *every* LLM output shown

Each *active agent* in a step makes up to four LLM calls, in this order. **All four are real model
outputs** (the broker is itself an LLM call):

```
① [REFINE persona]  reactive subscription — the agent rewrites its own system prompt for THIS question
② [ReAct RETRIEVE]  (knowledge roles only) the agent emits "SEARCH: <kw>"; Wikipedia obs is fed back
③ [PUBLISH]         the agent's actual message (its answer / analysis / code)
④ [BROKER LLM]      the agent, acting as broker, writes a description of the NEEDED next role …
   [BROKER MATCH]   … which is embedded and cosine-matched to real agents → the routed receivers
```

Outputs below are real, lightly trimmed (`…`). Full untrimmed traces:
`result/full_{gsm8k,mmlu,humaneval}.txt`; raw JSON in `logs/`.

### 3.1 GSM8K — serial chain (`top_k=1`)

> **Task:** A company bought \$400,000 of equipment; 40% was faulty and returned. How much was spent
> on functioning equipment? **GOLD = 240000**, `active_per_step=[1,1,1]`

```
STEP 1  active = [Mathematical Analyst]
  ① REFINE   "You are a mathematical analyst tasked with calculating the total spent on functioning
              equipment… total $400,000… compute 40% faulty… subtract from the total…"
  ③ PUBLISH  "240000"
  ④ BROKER   "As the downstream agent I will assess the analysis provided… verify the correctness of
              the calculations and logic… present a step-by-step explanation of the reasoning…"
     MATCH   → routes to [Inspector]          # broker chose a verifier role, matched to Inspector

STEP 2  active = [Inspector]
  ① REFINE   "You are an Inspector… confirm the previous analyst's result of $240,000 and provide
              your own step-by-step process… output only the numerical result…"
  ③ PUBLISH  "1. Total = $400,000  2. Faulty = 40%  3. Faulty $ = 400,000×0.40 = 160,000
              4. Functioning = 400,000 − 160,000 = 240,000"
  ④ BROKER   "The most suitable downstream role is the Mathematical Analyst… calculate the total
              spent on functioning equipment by subtracting faulty value…"
     MATCH   → routes to [Mathematical Analyst]

STEP 3  active = [Mathematical Analyst]
  ① REFINE   "…follow these steps: identify total $400,000; faulty = total × 40%; subtract…"
  ③ PUBLISH  "240000"

FINAL   active = [Final Answerer]
  ① REFINE   "You are a mathematical analyst… determine the faulty amount then subtract from total…"
  ③ PUBLISH  "…Faulty = 400,000×0.40 = 160,000; Functioning = 240,000.  The answer is 240000"   ✓
```
*Solver → independent Inspector re-derives the arithmetic → solver confirms → final. Note the broker
LLM (④) each time writes a description of the *needed* role, which the embedding step resolves to a
concrete agent.*

### 3.2 MMLU — genuine fan‑out (`top_k=3`)

> **Task:** Who is an ‘injured State’ in the law of international responsibility? (4 options)
> **GOLD = D**, roster = the 9 MMLU roles, `active_per_step=[1,3]`

```
STEP 1  active = [Knowledge Expert]                                   # single entry host
  ① REFINE   "You are a legal expert specializing in international law, particularly state
              responsibility… evaluate each option against the definitions/principles…"
  ③ PUBLISH  "…analyze each option against the ILC Articles on State Responsibility. Option A:
              injury is not solely 'damage'…"   (frames the problem, no commit yet)
  ④ BROKER   "I am a Lawyer with expertise in law, policy, and institutional reasoning. I analyze
              legal principles, definitions, and precedents… legality, compliance, doctrinal
              consistency…"                       # broker describes the needed role = a Lawyer
     MATCH   → routes to [Lawyer, Critic, Mathematician]   # FAN-OUT: top-3 nearest subscriptions

STEP 2  active = [Mathematician, Critic, Lawyer]            # three heterogeneous roles, in parallel
  Mathematician:
    ① REFINE  "You are a legal expert in international law… evaluate the definition of an 'injured
               State'… reference the ILC Articles… under 100 words."   # ← even the Mathematician
                                                                        #   re-specializes to law!
    ③ PUBLISH "Answer D. An injured State = an obligation breached owed to it individually or to a
               group incl. it, and it was specially affected. A/B/C incomplete."
  Critic:
    ① REFINE  "You are a legal expert in international law focusing on 'injured State'… critically
               evaluate the options per the ILC Articles…"
    ③ PUBLISH "A incorrect (only damage); B too narrow (peremptory norms); C misleading
               (acknowledgment ≠ injured); D correct, aligns with ILC Articles."
  Lawyer:
    ① REFINE  "You are a lawyer specializing in state responsibility… assess each option for
               doctrinal consistency…"
    ③ PUBLISH "Answer D. A relevant but insufficient; B too narrow; C wrong; D accurately reflects
               the ILC criteria (individually owed / specially affected)."

FINAL   active = [Final Answerer]
  ① REFINE   "You are a legal expert in international law… decide the single correct option…"
  ③ PUBLISH  "D"   ✓
```
*The real RAPS design: the broker LLM (④) diagnoses “we need a lawyer”, the embedding step fans that
out to the 3 nearest roles (incl. the Critic), and **reactive subscription (①) re-specializes every
one of them — even the Mathematician — into a legal expert** for this question. They deliberate
independently and converge on D. Contrast `top_k=1`, where the same question type degenerates into a
5-step echo chamber of two roles.*

**Same question with ReAct retrieval ON** (a knowledge agent, post kstem+keyword fix) adds step ②:
```
  ② RETRIEVE  SEARCH "injured State international responsibility definition"
              OBS → [State responsibility] "…states may owe secondary obligations…"   (clean hit)
  ③ PUBLISH   "…based on the precise definition retrieved: D"   ✓
```

### 3.3 HumanEval — design → implement → test → repair (`top_k=1` + `code_verify`)

> **Task:** `do_algebra(operator, operand)` — build & evaluate an expression from two lists
> (`+ - * // **`). `active_per_step=[1,1,1,1,1]`

```
STEP 1  active = [Project Manager]
  ① REFINE   "You are a project manager specializing in concise code for algebraic expression
              evaluation… a single function processing the operator/operand lists…"
  ③ PUBLISH  "…iterate through operands applying operators in order. Here's the implementation:
              ```python def do_algebra(...): ...```"
  ④ BROKER   "As a Bug Fixer I will analyze the signature and docstring, identify and correct issues
              … provide a complete modified implementation…"
     MATCH   → routes to [Bug Fixer]

STEP 2  active = [Bug Fixer]
  ① REFINE   "You are a bug fixer… enhance do_algebra… follow order of operations… handle all
              operators… keep the signature…"
  ③ PUBLISH  "```python def do_algebra(operator, operand): … ```"   (full implementation)
  ④ BROKER   "The most suitable downstream role is Test Analyst… identify potential issues, special
              use cases, boundary conditions… empty lists, unsupported operators…"
     MATCH   → routes to [Test Analyst]

STEP 3  active = [Test Analyst]
  ① REFINE   "You are a test analyst focused on evaluating do_algebra… identify potential failures…"
  ③ PUBLISH  "Operator precedence is ignored; division by zero unhandled; validate empty lists;
              consider negatives / single-operator cases."
  ④ BROKER   → routes to [Bug Fixer]

STEP 4  active = [Bug Fixer]
  ③ PUBLISH  "```python …revised implementation addressing the feedback… ```"
  ④ BROKER   → routes to [Test Analyst]

STEP 5  active = [Test Analyst]
  ③ PUBLISH  "Residual edge cases: 0**0, negative operands."

FINAL   active = [Final Answerer]
  ③ PUBLISH  "```python …final code block… ```"
  → code_verify runs the public doctests on this code; if any fail it repairs (≤N iters) and re-checks
    before returning.   (This objective test loop is what makes HumanEval gains safe.)
```
*The closed loop **designer → coder → tester → fixer**, plus the final **doctest execution + repair**,
lifts HumanEval 0.781 → ~0.887. The objective test oracle is the key: a wrong fix can’t pass the
tests, so breakage ≈ 0.*

---

## 4. Retrieval Infrastructure (Elasticsearch + BM25 + ReAct)

- **Index `wiki_kstem`** — 27,185,091 passages, 4 shards, custom analyzer (`standard` tokenizer +
  `lowercase` + **`kstem`** light stemmer). Built via `_reindex` from the original `wiki` index.
- **Why `kstem` and not the default `english` analyzer:** the built‑in `english` (Porter) stemmer
  **over‑stems** — `international`, `internal`, `internment`, `interned` all collapse to `intern`,
  poisoning retrieval (a query about *international law* matched *internment*/*internal* passages).
  `kstem` keeps those distinct while still normalizing plurals/inflections
  (`responsibilities→responsibility`, `injuries→injury`). The old index baked the bad stemming in,
  so a **full re‑index was required**.
- **Query construction matters more than the index.** A natural‑language query
  (`"Definition of 'injured State' in international law"`) dilutes the signal across common terms and
  OR‑matches junk (top hit was *Weapon of mass destruction*). A **keyword** query
  (`"injured State responsibility breach obligation specially affected"`) cleanly returns
  *State responsibility*. The ReAct prompt now forces keyword‑style queries.
- `bm25_retrieve(queries, k=5, chars=500)` — text‑only `match` (no title boost; a title^2 boost once
  matched “Injured State” → baseball “Injured list”). Index overridable via `WIKI_INDEX` env.

---

## 5. What Works / What Doesn’t (honest, all A/B verified)

| Lever | Dataset | Net effect | Why |
|---|---|---|---|
| `code_verify` + Edge‑Case Writer | HumanEval | **+10.6pp → 0.887** ✅ | objective test oracle → safe repair, 0 breakage |
| PAL `tool_verify` | GSM8K | **+0** | apparent recovery was temperature re‑rolls, not the tool; baseline already 0.922 |
| Multi‑agent fan‑out (top_k=3) | MMLU | ~+0 to slightly − | homogeneous roles echo each other; lowering threshold pulls in irrelevant roles |
| **ReAct Wikipedia retrieval** | MMLU | **+0 (76=76)** | retrieves the *right* article, but only helps fact/definition Qs; breaks an equal number where context misleads; most failures need *reasoning*, not facts |
| CoT depth | MMLU | **+0 (0.70=0.70)** | failures are confident *wrong‑frame* reasoning; more CoT elaborates the wrong frame |
| Arbitration routing (4 variants) | MMLU | +0 to +1 | the only available discriminator is the same model, ~chance on hard disagreements |

**MMLU verdict:** with `gpt-4o-mini` and no backbone change, MMLU sits at ~0.76–0.84 (sample
dependent). An *oracle* upper bound (perfect routing across panel/few‑shot/code) is ~0.89, but it is
**not realizable** because the discriminator is the same fallible model. The dominant failure class
is **professional_law** and other reasoning/application questions where the model is *confidently
wrong* — not fixable by retrieval, CoT, or coordination. This is a capability ceiling, not a bug.

---

## 6. Bugs Found & Fixed (and one to watch)

| Bug | Impact | Fix |
|---|---|---|
| Broker degenerated MMLU to `top_k=1` chain | the “multi‑agent” baseline was a 5‑step echo chamber of 2 roles | run with `top_k=3` (now the default in `run_mmlu.py`) |
| ES `english` analyzer over‑stemming (`international→intern`) | poisoned retrieval | re‑index as `wiki_kstem` (kstem) |
| ReAct agents wrote natural‑language queries | BM25 returned off‑topic junk | prompt forces **discriminative keyword** queries |
| BM25 `title^2` boost | “Injured State” → baseball “Injured list” | text‑only match |
| `execute_code_get_return` used separate globals/locals | imports failed inside functions | single namespace `exec(code, ns)` |
| doctest leaked closing `"""` into last `want` | false code_verify failures | `_clean_want()` |
| **Log filename collision** (`log_mmlu_{ts}_{slug}`, slug = task[:10]) | questions sharing a 10‑char prefix (“Which of the…”, “This question…”) **overwrote** each other → 227 runs produced only 172 logs; biased all per‑question log analysis | **fixed:** filename now includes the question index `…_{i:05d}_{slug}.json`; `write_task_log(…, index=i)`. *Headline accuracies came from the live counter and were unaffected, but re‑derive any per‑question analysis from the new index‑tagged logs.* |
| Self‑kill: `pkill -f <script>.py` in the same shell command that launches `<script>.py` | kills the launcher | launch standalone (`setsid nohup … </dev/null &`), kill by PID in a separate command |

---

## 7. How to Run / Reproduce Key Results

```bash
conda activate raps
export RAPS_TEMPERATURE=0 no_proxy='*' NO_PROXY='*'

# MMLU — current design (top_k=3 fan-out + ReAct retrieval, kstem index)
python experiments/run_mmlu.py --limit 100 --start 0 --top_k 3 --sim_threshold 0.40 --max_steps 5
# MMLU — clean baseline (chain, no retrieval) for ablation
python experiments/run_mmlu.py --limit 100 --start 0 --top_k 1 --sim_threshold 0.40 --max_steps 5 --no_react

# GSM8K / HumanEval
python experiments/run_gsm8k.py     --limit 200
python experiments/run_humaneval.py --limit 160   # code_verify + edge_cases enabled in config

# Retrieval sanity
python -c "import os; os.environ['no_proxy']='*'; from RAPS.tools.search.bm25_retriever import bm25_retrieve; \
print(bm25_retrieve(['injured State responsibility breach obligation specially affected'], k=5))"
```

Per‑question logs land in `logs/log_<dataset>_<timestamp>_<index>_<slug>.json` (now collision‑free).
Each log contains `task`, `team` (roster, top_k, active_per_step), and per‑step `publications`,
`refinements`, `broker_decisions`, `llm_calls` (incl. `retrieve` traces), `reputation`.

---

## 8. File Map (the parts that matter)

```
RAPS/core/coordinator.py          # RAPSCoordinator + RAPSConfig (execution loop, final decision, tool/code verify)
RAPS/graph/node.py                # broker_route, _match_by_embedding, refine_system_prompt, watchdog/reputation
RAPS/agents/analyze_agent.py      # MMLU agent + ReAct retrieval loop (_react_gen)
RAPS/agents/math_solver.py        # GSM8K agent
RAPS/agents/code_writing.py       # HumanEval agent
RAPS/agents/retriever_agent.py    # standalone retrieval host (legacy / optional)
RAPS/agents/seed_pool.py          # recruitable specialists per domain
RAPS/prompt/{mmlu,gsm8k,humaneval}_prompt_set.py   # roles, role descriptions, constraints
RAPS/llm/azure_chat.py            # gpt-4o-mini gateway + embedding client
RAPS/tools/search/bm25_retriever.py   # local Wikipedia BM25 (index = wiki_kstem)
RAPS/tools/coding/python_executor.py  # PAL / code execution
serving/embedding_server.py       # Qwen3-Embedding-4B service (GPU0, :8200)
experiments/run_{mmlu,gsm8k,humaneval}.py   # benchmark drivers
experiments/mmlu_*.py             # MMLU ablations (recovery, router, fewshot, design, sweep)
logs/                             # per-question JSON traces
result/                           # run stdout logs + case_*.txt dumps
```

---

## 9. Open Threads / Suggested Next Steps

1. **MMLU is capability‑bound, not engineering‑bound.** Further design tweaks (more agents, more
   retrieval, more CoT) have repeatedly netted ≈0. The honest path to higher MMLU is a stronger
   backbone, not more coordination.
2. If retrieval is pursued further, the bottleneck is **passage‑level precision** for hyper‑specific
   detail questions — try **Qwen3 semantic (vector) retrieval** or a reranker instead of BM25, and
   suppress retrieval on no‑fact subjects (e.g. `moral_scenarios`) where it only adds noise.
3. The **reputation/robustness** machinery is implemented but lightly evaluated — `run_robustness.py`
   with an injected `AdversarialAgent` is the place to demonstrate the paper’s robustness claims.
4. Re‑run any **per‑question** MMLU analysis on the new index‑tagged logs (the old 172‑log set was
   biased by the filename‑collision bug).

---

## 10. Setting up the excluded large assets (NOT in git)

The repo intentionally **excludes** the embedding model, the Wikipedia dump/index, and two oversized
dataset blobs (see `.gitignore`). The **run‑result JSON traces (`logs/`, `result/`) ARE committed.**
Recreate the excluded pieces as follows.

### 10.1 Embedding model — `models/Qwen3-Embedding-4B/` (~8 GB)

Used by the broker for routing (replaces `text-embedding-3-small`, which the gateway doesn’t expose).

```bash
conda activate embed            # Python 3.12, torch 2.4.1+cu121 (box driver is CUDA 12.2)
export HF_HUB_ENABLE_HF_TRANSFER=1 no_proxy='*' NO_PROXY='*'
huggingface-cli download Qwen/Qwen3-Embedding-4B --local-dir models/Qwen3-Embedding-4B
# start the OpenAI-compatible service on GPU 0, port 8200
nohup bash serving/start_embedding.sh > serving/embed_server.log 2>&1 &
curl -s --noproxy '*' localhost:8200/health     # -> {"status":"ok","dim":2560,...}
```

### 10.2 Wikipedia dump + Elasticsearch index — `wiki/` (11 GB) → index `wiki_kstem` (27.2 M passages)

Used by the ReAct retrieval. Download the dump, bulk‑index BM25, then re‑index with the **kstem**
analyzer (the default `english` analyzer over‑stems — see §6).

```bash
conda activate raps
export no_proxy='*' NO_PROXY='*'

# (a) download the 2023-11 English Wikipedia parquet dump into wiki/20231101.en/
huggingface-cli download wikimedia/wikipedia 20231101.en --repo-type dataset \
    --local-dir wiki/20231101.en

# (b) start Elasticsearch 8.15 (single-node, security off, bundled JDK; geoip downloader off, no_proxy)
#     ES_JAVA_HOME must point at the bundled JDK, NOT the env's jdk8.

# (c) build the base BM25 index `wiki` (chunks articles into ~128-word passages; takes a while)
python RAPS/tools/search/index_wiki.py

# (d) re-index into `wiki_kstem` with the light kstem stemmer (this is what the retriever uses)
python - <<'PY'
import os; os.environ['no_proxy']='*'
from elasticsearch import Elasticsearch
es=Elasticsearch("http://127.0.0.1:9200", request_timeout=120)
es.indices.create(index="wiki_kstem", body={
  "settings":{"index":{"number_of_shards":4,"number_of_replicas":0,"refresh_interval":"-1"},
    "analysis":{"analyzer":{"default":{"type":"custom","tokenizer":"standard",
                 "filter":["lowercase","kstem"]}}}},
  "mappings":{"properties":{"title":{"type":"text"},"text":{"type":"text"}}}})
es.reindex(body={"source":{"index":"wiki","size":2000},"dest":{"index":"wiki_kstem"}},
           slices="auto", wait_for_completion=True, request_timeout=3600)
es.indices.put_settings(index="wiki_kstem", body={"index":{"refresh_interval":"1s"}})
print("done:", es.count(index="wiki_kstem")['count'])
PY
```
The retriever reads `WIKI_INDEX` (default `wiki_kstem`): `RAPS/tools/search/bm25_retriever.py`.

### 10.3 Oversized dataset blobs (excluded, not needed for eval)

`raps_data/MMLU/data.tar` (159 MB, a redundant tarball of `data/`) and
`raps_data/MMLU/data/auxiliary_train/race.csv` (147 MB, MMLU auxiliary **training** set) exceed
GitHub’s 100 MB limit and are not used for evaluation. The MMLU **test/dev** CSVs that the
benchmarks actually read *are* committed. If you need the auxiliary set, re‑download the MMLU data
bundle and extract it under `raps_data/MMLU/`.

## Towards Adaptive, Scalable, and Robust Coordination of LLM Agents:<br>A Dynamic Ad-Hoc Networking Perspective

<img src="./image/raps.png" width="100%" alt="RAPS Overview"/>

[[📄]](https://arxiv.org/abs/2502.xxxxx) _Towards Adaptive, Scalable, and Robust Coordination of LLM Agents: A Dynamic Ad-Hoc Networking Perspective_

Multi-agent architectures built on large language models (LLMs) have demonstrated the potential to realize swarm intelligence through well-crafted collaboration. However, the substantial burden of manual orchestration inherently raises an imperative to automate the design of agentic workflows. We frame such an agent coordination challenge as a classic problem in dynamic ad-hoc networking: _How to establish adaptive and reliable communication among a scalable number of agentic hosts?_ In response to this unresolved dilemma, we introduce **RAPS**, **a Reputation-Aware Publish-Subscribe paradigm for adaptive, scalable, and robust coordination of LLM agents**. RAPS grounds its ad-hoc coordination fabric in a Distributed Content-Centric Protocol, allowing agents to exchange messages based on their declared intents rather than predefined communication topologies. Beyond such a flexible substrate, RAPS further incorporates two critical overlay mechanisms: (i) Reactive Subscription, which enables agents to refine their intents on the fly; and (ii) Bayesian Reputation, which empowers agents with a local watchdog to isolate malicious peers. Extensive experiments on five benchmarks showcase that RAPS effectively reconciles adaptivity, scalability, and robustness within a unified coordination framework.

### 🛠️ Requirements
- python >= 3.9
- openai == 2.16.0
- numpy == 2.4.1
- scipy == 1.17.0
- tenacity == 9.1.2
- class-registry == 2.1.2
- tiktoken == 0.12.0
- sentence-transformers == 5.2.2
- shortuuid == 1.0.13
- wikipedia == 1.4.0
- datasets

### 📚 Datasets

| Dataset | Task | Access |
|----------|------|--------|
| **GSM8K** | Math Reasoning | [HuggingFace](https://huggingface.co/datasets/gsm8k) |
| **MMLU** | General Knowledge | [HuggingFace](https://huggingface.co/datasets/cais/mmlu) |
| **HumanEval** | Code Generation | [HuggingFace](https://huggingface.co/datasets/openai_humaneval) |

The datasets can be automatically downloaded and prepared via the `datasets` library, or alternatively loaded from local files using the provided scripts.

### 🚀 Quick Start

#### 1. Setup Environment
```bash
git clone https://github.com/your-repo/RAPS.git
cd RAPS-main
pip install -r requirements.txt
export OPENAI_API_KEY="... (openai key)"
```
Alternatively, the API key can also be configured in `RAPS/llm/openai_key.txt`. Please note that the LLM calls would incur API costs.

#### 2. Run a Quick Demo (GSM8K Example)
```bash
python experiments/run_gsm8k.py --llm_name gpt-4o-mini --max_steps 3
```
This command launches a minimal end-to-end demo on GSM8K: it initializes the agent group, runs multi-step coordination, and saves detailed logs (including prompts, responses, and broker decisions) to `./logs/`.

### 📊 Output
- **Logs**: Detailed execution logs (prompts, responses, broker decisions) are saved in the `logs/` directory with timestamps (e.g., `log_gsm8k_2026-02-01_....json`).
- **Results**: Final metrics (Accuracy, Pass@1) are printed to the console and saved in the `result/` directory.

## 📜 Acknowledgement
This project builds on insights from dynamic network theory, multi-agent systems, and large language model research. We thank the authors of GSM8K, MMLU, and HumanEval for their benchmarks.

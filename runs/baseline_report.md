# SWE-bench Lite — 单智能体基线报告 (Single-Agent Baseline)

**用途**:作为 RAPS 多智能体方案的**参照基线**。同一套固定 20 题、同一模型,后续 RAPS 与此对比。
**日期**:2026-06-14

## 0. 结论 (TL;DR)

| 指标 | 值 |
|---|---|
| **resolved (解决率)** | **2 / 20 = 10.0%** |
| 解决的题 | `django__django-10914`, `django__django-11039` |
| 出补丁 | 16/20(4 题空补丁) |
| 评测出错 | 2(`astropy-12907`, `psf__requests-1963`,补丁应用/评测失败) |
| 总成本 | **$0.78** | 
| 总 token | 5.0M | 
| 平均轮数 | 22.9 | 
| 总耗时(墙钟,4 worker) | 11.7 min |

## 1. 配置 (Stack)

| 项 | 值 |
|---|---|
| 骨架 | **mini-swe-agent 2.4.1**(单智能体 bash 循环) |
| 模式 | **backticks + `litellm_textbased`**(文本解析;默认 tool-calling 模式对 gpt-4o-mini 不可靠→`RepeatedFormatError`) |
| LLM | **gpt-4o-mini** @ ChatAnywhere(OpenAI 兼容网关) |
| 评测引擎 | **swebench 4.1.0 harness** + Docker(官方预构建镜像,`-n swebench`) |
| 关键参数 | `step_limit=250`, `cost_limit=$3`, `pull_timeout=1800`, workers=4 |
| 单价 | gpt-4o-mini:输入 $0.15/1M,输出 $0.60/1M |
| token 口径 | "billed"(每轮把历史重发,累计计);tiktoken `o200k_base` 估算,与 litellm 实测成本误差 ~0.5% |

**复现命令**:
```bash
cd /home/lirui/raps_swe/RAPS-main
# 1) 跑 agent + 采集指标
sg docker -c 'HF_ENDPOINT=https://hf-mirror.com /home/lirui/anaconda3/envs/raps_swe/bin/python \
  runs/run_baseline_metrics.py runs/eval20.txt runs/baseline_v2 4'
# 2) 评 resolved
sg docker -c "HF_ENDPOINT=https://hf-mirror.com /home/lirui/anaconda3/envs/raps_swe/bin/python \
  -m swebench.harness.run_evaluation -d princeton-nlp/SWE-bench_Lite -s test \
  -p runs/baseline_v2/preds.json -i \$(paste -sd' ' runs/eval20.txt) -id baseline_v2 \
  --max_workers 4 --cache_level env --clean True -n swebench"
```

## 2. 评估集 (固定 20 题,12 仓库分层抽样)

`runs/eval20.txt`:django×5, sympy×3, matplotlib×2, scikit-learn×2, pytest×1, sphinx×1, astropy×1, requests×1, pylint×1, xarray×1, seaborn×1, flask×1。

## 3. 每题明细 (resolved / rounds / tokens / cost / time)

resolved 图例:✅ 解决 · ❌ 未解决 · ⬚ 空补丁 · ⚠️ 评测出错

| instance | resolved | 轮数 | total tokens | USD | 墙钟(s) | 出补丁 |
|---|:--:|---:|---:|---:|---:|:--:|
| django__django-10914 | ✅ | 12 | 59,155 | 0.0095 | 37 | ✓ |
| django__django-10924 | ❌ | 22 | 187,141 | 0.0291 | 76 | ✓ |
| django__django-11001 | ❌ | 20 | 160,480 | 0.0249 | 65 | ✓ |
| django__django-11019 | ❌ | 21 | 211,002 | 0.0326 | 68 | ✓ |
| django__django-11039 | ✅ | 6 | 17,809 | 0.0030 | 16 | ✓ |
| sympy__sympy-11400 | ⬚ | 15 | 143,246 | 0.0224 | 140 | ✗ |
| sympy__sympy-11870 | ⬚ | 16 | 148,917 | 0.0233 | 103 | ✗ |
| sympy__sympy-11897 | ❌ | 25 | 295,447 | 0.0459 | 154 | ✓ |
| matplotlib__matplotlib-18869 | ❌ | 17 | 120,617 | 0.0190 | 60 | ✓ |
| matplotlib__matplotlib-22711 | ⬚ | 15 | 69,402 | 0.0115 | 67 | ✗ |
| scikit-learn__scikit-learn-10297 | ❌ | 23 | 178,931 | 0.0281 | 74 | ✓ |
| scikit-learn__scikit-learn-10508 | ❌ | 6 | 24,341 | 0.0040 | 54 | ✓ |
| pytest-dev__pytest-11143 | ❌ | 45 | 949,609 | 0.1455 | 234 | ✓ |
| sphinx-doc__sphinx-10325 | ❌ | 37 | 538,501 | 0.0832 | 377 | ✓ |
| astropy__astropy-12907 | ⚠️ | 16 | 95,418 | 0.0155 | 55 | ✓ |
| psf__requests-1963 | ⚠️ | 29 | 167,487 | 0.0274 | 108 | ✓ |
| pylint-dev__pylint-5859 | ❌ | 33 | 441,315 | 0.0677 | 108 | ✓ |
| pydata__xarray-3364 | ❌ | 29 | 363,262 | 0.0563 | 111 | ✓ |
| mwaskom__seaborn-2848 | ❌ | 47 | 642,456 | 0.0997 | 453 | ✓ |
| pallets__flask-4045 | ⬚ | 24 | 202,952 | 0.0315 | 74 | ✗ |

## 4. 汇总 (Summary)

| 指标 | 值 |
|---|---|
| 题数 | 20 |
| **resolved 解决率** | **2/20 = 10.0%** |
| 出补丁 | 16/20 |
| 空补丁 | 4(sympy-11400, sympy-11870, matplotlib-22711, flask-4045) |
| 评测出错 | 2(astropy-12907, requests-1963) |
| 总 token | 5,017,488 (~5.0M) |
| 总成本 | **$0.78**(litellm 实测 $0.7804) |
| 平均轮数 | 22.9(范围 6–47) |
| 总耗时(墙钟,4 worker) | 701s ≈ 11.7 min |
| 平均每题成本 | ~$0.039 |
| 平均每题 token | ~250k |

## 5. 观察 (Observations)

- **解决的 2 题都是 django**(配置类小改动:`django-10914` 改默认权限、`django-11039` 迁移 SQL);轮数都很少(12、6)。
- **空补丁 4 题**:agent 跑到 step/逻辑尽头没产出 diff(sympy×2、matplotlib、flask)。
- **评测出错 2 题**:补丁非空但 harness 应用/跑测试失败(很可能 diff 不干净)。后续可在 agent 端加"补丁能否 apply"的自检。
- 成本/轮数差异极大:pytest-11143 烧到 $0.146/45 轮/95 万 token 仍未解决 → 弱模型在难题上空转。

## 6. 下一步 (Next: RAPS)

RAPS 多智能体方案将在**同一 20 题**上对比本基线(目标:用发布订阅 + 反应式订阅 + 声誉协调多个 agent-loop,提升 resolved%)。
关键对比维度:**resolved% / 平均轮数 / 每题成本 / 耗时**。

> 数据源:`runs/baseline_v2/metrics.json`、轨迹 `runs/baseline_v2/<id>/<id>.traj.json`、resolved 报告 `gpt-4o-mini.baseline_v2.json`。

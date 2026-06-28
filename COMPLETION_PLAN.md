# RAPS 完善方案(Adaptive Agent Team)

目标:在保持论文 RAPS 三机制(发布/订阅、反应式订阅、贝叶斯声誉)的前提下,把当前"半成品"
补全为一个**自适应 agent team**——团队的成员、规模、路由、轮数都随任务与声誉动态变化,
并能复现论文的鲁棒性实验。LLM 统一走 Azure/modelhub 网关(gpt-4o-mini-2024-07-18)。

原则:一步一步来,每个 Phase 自成一个可运行、可验证的增量,改完即可冒烟测试。

---

## 现状诊断(已通读代码)

- **真身**在 `experiments/run_{gsm8k,mmlu,humaneval}.py` 的主循环里,不在 `graph.py`。
- 已实现:发布(`publish`)、broker 路由(`broker_route`,embedding 语义匹配)、反应式订阅
  (`refine_system_prompt`)、watchdog(`watchdog_evaluate`)、一手声誉(`update_first_hand`)。
- **缺口**:
  1. 声誉"只算不用"——`check_trust` 从不调用,二手声誉 `update_from_report` 完全闲置,
     声誉跨任务不重置 → 论文核心"隔离恶意节点"没有闭环。
  2. team 是**固定 4 角色全程在场**,没有"自适应"——不会按任务激活/休眠/招募 agent。
  3. 终止条件 `all_ready` 找的是 `"final answer: yes"` 字符串,agent 从不输出 → 死逻辑,永远跑满 max_steps。
  4. 大量死代码:`graph.py`(引用未定义的 GCN/MLP,依赖 torch/torch_geometric,实验没用)、
     `subscription.py`、`seed_pool.py`;三份 run_*.py 主循环重复。
  5. 实验脚本 bug:gsm8k `dataset[54:100]` 硬编码、`entry_agent` 不一致、`float(pred)` 无容错等。
  6. `adversarial_agent.py` 没接入 → 无法复现鲁棒性实验。

---

## Phase 0 — 打通 Azure 网关 + 冒烟跑通(基础设施)

**做什么**
- 新增 `RAPS/llm/azure_chat.py`:`AzureGPTChat`,用 `AzureOpenAI`:
  - `azure_endpoint = https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl`(办公网域名)
  - `api_version="2024-02-01"`,`default_headers={"X-TT-LOGID": <生成的 logid>}`
  - 默认 `model="gpt-4o-mini-2024-07-18"`;key 从环境变量/本地文件读,**不入库**。
- `LLMRegistry.get`:当 llm_name 形如 `gpt-4o-mini*` 时返回 `AzureGPTChat`,保留旧 `GPTChat` 兼容。
- **embedding 改造**:modelhub 网关不一定提供 embeddings 接口 → broker 的语义匹配改用本地
  `sentence-transformers`(已在 requirements),去掉对 OpenAI/Azure embedding 的硬依赖;
  `Node._match_by_embedding` 与 `cosine_similarity` 统一走它。
- `requirements.txt`:把 torch / torch_geometric 标为可选(仅 legacy 用),确认 sentence-transformers 真用上。

**验收**:`python experiments/run_gsm8k.py --llm_name gpt-4o-mini-2024-07-18 --max_steps 2`,前 3 题能产出答案与日志。

---

## Phase 1 — 抽出统一协同内核 + 清理死代码

**做什么**
- 新建 `RAPS/core/coordinator.py`:把三份 run_*.py 的主循环抽成 `RAPSCoordinator`,
  参数化(domain、entry 策略、数据切片、max_steps、top_k、阈值)。三个实验脚本只做
  "建 agent + 准备数据 + 评估指标",循环复用内核。
- 修 bug:`entry_agent` 统一策略;`float(pred)==float(answer)` 加 try/except;数据切片走参数;
  删掉假的 `all_ready` 字符串逻辑(终止条件在 Phase 3 用真实信号替换)。
- 死代码隔离:`graph.py` / `subscription.py` / `seed_pool.py` / 仅服务死代码的 `profile_embedding`
  移到 `RAPS/legacy/` 并在 README 注明,保证 `import RAPS` 不再触碰崩溃路径。

**验收**:三个实验脚本都跑通且行为与改前一致(回归),代码去重。

---

## Phase 2 — 闭环贝叶斯声誉(robustness 核心)

**做什么**
- **路由门控**:broker 选出接收方后,用 `check_trust(sender)` 基于 `REP` 过滤——
  声誉低于阈值的发送方消息被丢弃/降权,真正"隔离恶意节点"。
- **二手声誉**:每轮末 agent 间交换 `export_first_hand()` → `update_from_report()`(gossip),
  让信任在团队内传播(CONFIDANT 的 deviation test + 信任度更新)。
- **重置策略**:`reset_reputation_per_task` 开关,默认每题重置以对齐单题评测;另留"累积"模式。
- 把每轮 REP/TRUST 快照写入 `step_log`,便于分析与画鲁棒性曲线。

**验收**:日志能看到声誉随 watchdog 结果演化,且低声誉 agent 的消息被路由层拦下。

---

## Phase 3 — 自适应 Agent Team(本次重点)

把"固定 4 角色"升级为"按需自适应的团队":

- **(a) 动态成员**:不再每轮全员在场。下一轮 active agent 由 broker 语义匹配 **× 声誉权重** 决定,
  激活数随匹配分动态变化(team 自适应扩张/收缩),无人匹配的 agent 自动休眠。
- **(b) 动态招募(seed pool 接入)**:当现有团队对预测意图的 broker 匹配分都低于阈值时,
  从重写后的 `SeedAgentPool`(补全为各 domain 的有效角色+合法 LLM 名)实例化新角色 agent 加入团队
  (team 自适应"生长")。
- **(c) 自适应终止**:用真实收敛信号替代死逻辑——例如"多数 agent 数值答案一致"或
  "Inspector/Verifier 通过"即提前停止,省成本、降轮数。
- **(d) 容量自适应**:`max_steps` / `top_k` / 激活上限随任务难度(题面长度、历史分歧度)动态调整。

**验收**:同一批题里,不同题目展现出不同的团队规模/成员/轮数;日志可见招募与休眠事件。

---

## Phase 4 — 鲁棒性评测 harness

**做什么**
- 接入 `adversarial_agent.py`:可配置注入 N 个/比例的恶意 agent(输出幻觉、错误数值、对抗性误导)。
- 新增对比脚本:在不同恶意比例下,跑 {开/关 声誉门控},输出准确率对比,复现论文鲁棒性增益。

**验收**:能产出一张"恶意比例 vs 准确率(有/无声誉)"的对比结果。

---

## Phase 5 — 收尾

- 日志/结果分析脚本、README 更新(新架构、Azure 配置、运行指令)。
- 可选:MMLU / HumanEval 全量复现与超参整理。

---

## 执行顺序与依赖

Phase 0 → 1 是基础(先能跑、先收敛骨架);Phase 2 → 3 是论文价值核心(声誉闭环 + 自适应团队);
Phase 4 验证鲁棒性;Phase 5 收尾。每个 Phase 改完都做一次小样本冒烟测试再进入下一个。

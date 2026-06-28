# RAPS on SWE-bench — 实现设计方案 (Design)

**定位**:性能优先(resolved% 最大化,不优化成本)。

### 约束(更新)

- **唯一硬不变量**:保留 **RAPS 核心 = 动态 ad-hoc 自组网**——多轮;**每一轮不是单个 LLM 出动作,而是一次内容寻址的发布/订阅协调**(broker 路由 + 反应式订阅 + 贝叶斯声誉)。
- **其余一切 for 性能**:**允许改/弃 mini-swe 骨架**、自定义循环与每轮结构、任意 prompt/工具/调用次数;**不必与基线保持一致**。mini-swe 的 `DockerEnvironment` 仍**优先复用**(纯粹因为它的 SWE-bench Docker 管线已验证可靠 = 为性能服务,而非为对齐基线)。
- **仍保留的工程约束**:(1) 不影响其他数据集实验(SWE 代码全部隔离在 `RAPS/swe/`);(2) 同一 backbone **gpt-4o-mini**(科学对比:同模型下"单 agent vs RAPS 自组网",模型可换是另一个旋钮)。

> 基线参照(同 20 题,仅作 resolved% 对照点,**方法不必对齐**):单 agent mini-swe + gpt-4o-mini = **2/20 = 10%**,$0.78,平均 22.9 轮。

---

## 0. 核心思想:两级结构

```
外层(沿用 mini-swe-agent,多轮):  一个持久 Docker 容器(testbed@base_commit) + 共享 transcript(信息总线)
   run(): while not exit:  step() = execute_actions( query() )
                                              └────────────┬───────────┘
内层(RAPS 自组网,替换 query()):  每一轮不再是"单个 LLM 出动作",而是一次发布/订阅协调:
   感知状态 → 反应式订阅 → broker 路由(内容寻址)→ 候选发布 → watchdog+声誉 → 仲裁 → 选出本轮动作
```

**起点(可演进)**:子类化 `minisweagent.agents.default.DefaultAgent`,重写 `query()` —— 把"单 LLM 出动作"换成"一次 RAPS 网络轮",返回相同格式消息(`extra.actions=[…]`),从而**先复用**外层循环、容器、提交检测、限额、轨迹保存,快速跑通最小自组网。

**性能优先下允许的演进(不必守 query() 的"每轮一条命令"约束)**:一轮网络可产出**复合动作**(并行多候选补丁→测试择优)、可在一轮内做**多 persona 真并发**、可自定义 transcript 之外的结构化黑板/工件存储、可重写提交与回环逻辑。`DockerEnvironment` 作为可靠执行底座保留;其上的循环/每轮结构以性能为准自由设计。

每个 persona 调用**同一个** `LitellmTextbasedModel`(gpt-4o-mini),区别只在 **system persona(订阅意图)**——对应论文"共享 π_θ、各自 S_i"。

---

## 1. Agent 团队成员 (先定义团队)

团队 = 一组"host",每个持有一个**订阅(intent)**,broker 按内容把控制路由给最相关者。分两类:

### 行动者 (Actors,产出一条 bash 动作)

| 角色 | 订阅 / 意图(broker 匹配文本) | 典型动作 |
|---|---|---|
| **Localizer 定位者** | 从 issue 出发定位根因文件/函数;grep/find/ast 搜索;追踪符号、import、调用点 | 搜索、读文件 |
| **Reproducer 复现者** | 写一个最小脚本复现所报 bug,先跑确认它**失败**(建立失败基线) | 写+跑 repro 脚本 |
| **Editor 补丁作者** | 在定位处实现**最小、通用、与代码库一致**的源码修改;产出干净可应用的编辑 | 改源文件 |
| **Verifier 验证者** | 改动后跑 repro 脚本 + 仓库相关既有测试;报告通过/失败/回归;`git apply --check` | 跑测试 |

### 顾问 / 看门狗 (Advisor,不直接出动作,影响选择与终止)

| 角色 | 订阅 / 意图 | 作用 |
|---|---|---|
| **Reviewer 评审/看门狗** | 评审定位与补丁的正确性、通用性、回归风险;识别"错误框架"的修法与改测试文件的违规;判断可否提交 | watchdog 否决 + 仲裁投票 + 提交门控 |

### 可招募 (Adaptive,卡住时动态加入 —— 对应论文动态成员)

| 角色 | 订阅 / 意图 |
|---|---|
| **Deep Debugger 深度调试** | 进展停滞时提出**另一种根因假设/调试策略**(插桩打印、二分定位、查看运行期状态) |
| **Dependency/Import Specialist** | 报错集中在依赖/导入/环境时介入 |

> **去中心化**:没有中央 planner(论文明确反对单点)。Reviewer 是**对等节点**,影响力来自声誉与投票;**提交的硬门控是客观测试(Verifier 真跑),不是任何单个 agent 说了算** —— 规避单点失效。

---

## 2. 整体方案:一轮 RAPS 自组网 (query() 内部)

输入:共享 transcript `self.messages`、团队(personas+各自 `ReputationManager`)、容器 env。

1. **感知状态(发布到总线)**:从最近一次观测 + transcript 尾部,得到当前"需求"短描述
   (启发式/极小 LLM 调用):如 *"还没定位"* / *"已有补丁、未验证"* / *"测试在 X 处失败"*。这就是 broker 的路由 query。
2. **反应式订阅(Overlay I)**:候选 persona 把订阅/人设**特化到当前 issue+文件**
   (如 Editor → "本题 Django `get_order_by` 的修复专家")。**惰性刷新**(上下文实质变化才重算)以控成本。
3. **broker 路由(去中心化、内容寻址)**:用 embedding(`text-embedding-3-small`)把"需求"匹配到 persona 订阅
   → 选出本轮 active 集 R。默认 **|R|=1**(最贴合的专家);**自适应容量**:卡住/关键节点时 fan-out 到 k 个。
   **声誉门控**:跳过当前状态下被网络判为不可信的 persona。
4. **发布(Publication)**:每个 active persona 基于"共享 transcript + 特化人设"产出**一条候选**
   (思考 + 一个 ```mswea_bash_command``` 块,与基线同格式)。
5. **watchdog + 贝叶斯声誉(Overlay II)**:逐条核验候选命令——是否良构?**是否在改测试文件**(SWE-bench 禁止)?
   是否破坏性/离题?丢弃非法或被高声誉 Reviewer 否决者;更新一手声誉。
6. **仲裁**:若 >1 候选,按**声誉加权投票**(+ 可选 Reviewer 快判)选出本轮动作。
   **提交决策特殊**:仅当**客观 Verifier 已真跑通**(repro 通过、无明显回归)才允许 SUBMIT,不靠意见。
7. **返回所选候选**(格式同基类)→ 基类 `execute_actions` 在共享容器执行 → **观测追加进 transcript = 广播**给所有 persona(下一轮可见)。
8. **声誉更新**:依执行结果(returncode、是否推进、后续验证是否通过)更新行动 persona 的声誉;
   低声誉者后续被路由绕开 → 信息流远离不可靠节点(论文 Eq.10)。

> **典型轨迹是"涌现"的、非硬编码**:定位 → 复现 → 修改 → 验证;验证失败则该 Editor 动作声誉下降 →
> broker fan-out(双 Editor / 招募 Deep Debugger)→ 重改 → 重验(回环)。这正是"动态 ad-hoc 自组网"。

---

## 3. 三机制 ↔ 落地对照(论文一致性)

| 论文机制 | 这里的落地 |
|---|---|
| 分布式发布/订阅基底(可扩展) | 共享 transcript = 信息总线;broker 用 embedding 把"需求"匹配订阅,路由谁来行动;无固定拓扑 |
| 反应式订阅(自适应) | 每轮把 persona 特化到当前 issue/文件;卡住时**招募** seed 专家(动态成员) |
| 贝叶斯声誉 + watchdog(鲁棒) | 核验候选动作;按 persona 可靠度(其动作是否导致测试通过/被采纳)演化声誉;路由绕开低声誉者 |
| **客观验证(性能锚,HumanEval 教训)** | Verifier 真跑 repro+回归测试;**提交硬门控**;这是 resolved% 提升的主来源 |

---

## 4. 终止 & 针对基线失败的修正

基线失败:4 空补丁(放弃)、2 坏 diff(应用失败)、其余多为未解决。设计逐一修:

- **空补丁**:Reviewer + 可招募 Deep Debugger 让其不轻易放弃;且**临近 step/cost 上限时,只要已有任何编辑就提交 best-effort diff,绝不交空**。
- **坏 diff**:提交前 Verifier 必跑 `git apply --check` / 真应用 → 拦截畸形补丁。
- **未解决(错误框架)**:Reviewer 批判 + **Reproducer 的失败测试必须在补丁后真通过**(客观闸)→ 拦掉"看似对其实错"。
- **正常终止**:Verifier 确认 repro 通过且无明显回归 + Reviewer 通过 → 网络发 SUBMIT。

---

## 5. 代码布局 & 隔离(绝不影响其他数据集)

**全部新增在独立子包 `RAPS/swe/`,不改 QA 路径**:

```
RAPS/swe/__init__.py
RAPS/swe/team.py          # SWEPersona(角色/订阅/ReputationManager) + 团队定义 + seed 招募池
RAPS/swe/round.py         # 一轮 RAPS 自组网:感知/反应式订阅/broker/发布/watchdog/仲裁
RAPS/swe/agent.py         # SWERAPSAgent(DefaultAgent) —— 只重写 query()
RAPS/swe/run_swebench.py  # 批量 runner(产出 swebench 兼容 preds.json + 复用 run_baseline_metrics 的指标采集)
```

**复用(import,不修改)**:
- `RAPS/graph/reputation.py::ReputationManager`(声誉)
- `RAPS/graph/node.py::cosine_similarity` + LLM 后端 `get_embeddings`(broker 语义匹配)
- `RAPS/llm/gpt_chat.py`(ChatAnywhere,已就绪;**向后兼容**:仅当 `RAPS_LLM_BACKEND=GPTChat` 生效,QA 默认仍是 AzureGPTChat)
- `minisweagent`:`DockerEnvironment`、`DefaultAgent`、`LitellmTextbasedModel`、SWE-bench 镜像命名/提交检测

**不触碰**:`RAPS/core/coordinator.py`、`experiments/run_{gsm8k,mmlu,humaneval}.py`、所有 QA prompt/agent。
→ 其他数据集实验**零影响**(且本机 Azure 网关本就不可达,QA 路径与 SWE 路径互不依赖)。

---

## 6. 性能打法(性能优先,成本不设限,基线不必对齐)

既然解绑了"对齐基线/省成本",直接上 SWE-bench 已知最强杠杆,全部嵌进自组网:

- **复现测试为脊柱**:每题先让 Reproducer 建立一个**失败的 repro**;整条修复回环以"repro 由红转绿"为客观闸(无 repro 不轻易提交)。
- **多候选补丁 + 测试择优**(最大杠杆):关键修复轮 fan-out 出 **N 个** Editor 候选(不同策略/温度,**真并发**调用),各自在容器副本/`git stash` 隔离下应用,Verifier 跑 repro+回归,**按通过情况投票择优**——而非让单个 LLM 一次写对。
- **定位多路召回**:Localizer 同时用 LLM 推断 + embedding/grep 检索,合并取并,提高根因召回(基线常定位偏)。
- **真并发**:成本不敏感 → 同轮多 persona 并发出请求;broker/仲裁等齐再选。
- **放宽预算**:`cost_limit` 不设硬限(或 $1–2/题),`step_limit` 提高;Verifier 跑测试是 bash(LLM 便宜)但收益最大,放手跑。
- **声誉加速收敛**:连续失败的策略/persona 声誉下降被绕开,把预算导向有效路径。
- 预期成本数倍于基线,**刻意用成本换 resolved%**。

---

## 7. 评测与对比

同 `runs/eval20.txt` 跑 RAPS → `runs/raps_v1/`,同一 swebench harness 评 resolved%。
**对比维度**:resolved% / 平均轮数 / 每题成本($、token)/ 墙钟。复用 `runs/run_baseline_metrics.py` 的采集器
(每个 persona 调用计入轮数与 token)。

---

## 8. 实施顺序(增量,每步可验证)

1. `RAPS/swe/agent.py`:`SWERAPSAgent(DefaultAgent)`,`query()` 先只接 **1 个 actor + Reviewer watchdog**(最小自组网)→ 1 题冒烟,确认能产出可应用补丁。
2. 接 broker 路由(Localizer/Reproducer/Editor/Verifier 内容寻址)+ 反应式订阅。
3. 接 watchdog + 贝叶斯声誉门控 + 仲裁;接 Verifier 客观提交门控(+ 空/坏补丁兜底)。
4. 接自适应 fan-out + seed 招募(Deep Debugger)。
5. 跑全 20 题,与基线对比,出报告 `runs/raps_report.md`,做消融(去声誉/去反应式订阅/去 fan-out)。
```

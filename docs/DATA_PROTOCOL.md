# ProgFeed 数据协议

本协议对应 `student-sim-cd.progfeed.v1`。这是静态数据重建协议，不能作为判题重放、模型有效性或论文创新性的证据。

## 来源与运行

官方来源为 [ProgFeed 仓库](https://github.com/umass-ml4ed/progFeed-dataset-public) 及其 [数据字典](https://github.com/umass-ml4ed/progFeed-dataset-public/blob/main/DATA_DICTIONARY.md)。上游将数据标为 CC BY 4.0；使用时须保留上游归属，并按上游要求引用论文。本项目不将数据重新授予软件许可证，不将原始数据或处理后学生代码提交至 Git。

```sh
PYTHONPATH=src .venv/bin/python -m student_sim_cd.progfeed build \
  --source data/raw/progfeed \
  --output data/prepared/progfeed-v1 \
  --seed 20260929
```

输出目录必须不存在或为空。原始数据保持只读；构建不能覆盖原输出。`audit.json` 为所有读取的 CSV、代码、结果日志和题目文本记录路径、字节数及 SHA-256，并以排序清单生成整体内容指纹。下载时的仓库版本应由获取脚本另行记录。

## 样本和时间关系

CSV 的一行是一个函数或测试的判题记录，不是独立提交。先按 `(student_id, lab, submission_timestamp)` 聚合，并将 `all_labs/` 中未出现在 CSV 的提交目录纳入时间序列。时间格式固定为 `YYYY-MM-DD-HH-MM-SS`；不猜测时区。每个学生、每个作业的提交按时间排序。

一个样本预测相邻两次作业提交中**同一源文件的完整内容**。代码读取自 `all_labs/<lab>/<student>/<time>/<source_file>`，不是 CSV 的 `code_snippet`。读取 UTF-8/UTF-8 BOM 文本并保留原始换行；无法解码或缺失的代码单列报告。通过 CSV 文件名和提交目录下 `.py` 文件构建文件清单。

若相邻任一次提交缺少该文件，则该对进入 `excluded_pairs.jsonl`，不能越过中间提交与更晚记录配对。其中“下一提交没有同名文件”（可能改名或删除）与“原本预期存在的代码无法读取”分别标记。实际官方归档可见 `hello.py.py → hello.py` 的相邻改名；本协议不猜测改名映射，此类变化属于逐同名文件范围的排除项，不能误报为原始代码遗失。实际连续两次代码相同仍是有效样本，不作为重复记录删除。只删除完全相同的 CSV 行；同一测试键下不一致的重复结果保留并审计，不擅自选取一行。

历史 `H` 只包含同一学生、同一作业、同一源文件且严格早于当前提交的可读代码、当时结果和实际反馈。当前提交不重复进入 `H`，没有自动截断历史。缺失历史代码不补写，其数量进入审计。当前是同作业历史协议，不声称已利用该学生跨题的全部历史。

最后一次可见提交进入 `censored.jsonl`。无下一提交表示未观测到后续，不能解释成未修改、已掌握或学习完成。主样本因此以“存在可观测相邻下一提交”为条件，存在失访选择边界。

多文件作业仍保留逐文件样本，并统计涉及多文件的提交和样本数。当前输入不含其他伴随文件代码，输出也不是完整项目；这是明确的逐文件预测范围。后续完整项目模拟须另定表示与输入协议。多个文件和多次提交都不能作为独立学生用于统计显著性检验。

## 字段与物理隔离

`inputs.train.jsonl`、`inputs.dev.jsonl`、`inputs.test.jsonl` 仅包含：

| 字段 | 定义 |
|---|---|
| `schema_version` | 固定 `student-sim-cd.progfeed.v1` |
| `sample_id` | 当前学生、作业、源文件、当前时间的稳定散列，不依赖未来记录 |
| `student_id`, `lab`, `source_file`, `current_timestamp` | 当前样本身份 |
| `current_code` | 当前完整源文件文本 |
| `current_results` | 当前该文件的测试记录列表 |
| `history` | 按时间递增的 `{timestamp, code, results, feedback}` 列表 |
| `feedback` | 当前该文件实际非空下发的具体反馈列表 |
| `problem_statement` | 官方题目描述文本，缺失为 `null` |

每个测试记录为 `{test_name, function_name, status, score, max_score, testcase_mask}`。缺少或非法数字用 `null`，不以 0 代填；测试掩码必须是只含整数 0/1 的 JSON 数组，缺失或非法为 `null`，空数组保留但另计。没有记录的测试不是失败，也不是通过。分数不是逐测试通过率，不能据总分构造伪测试掩码。

每个反馈记录为 `{test_name, function_name, assigned_type, text}`。以 `ai_feedback_text` 非空判断实际下发；`assigned_type` 只是条件分配。`tc`/`nl` 行没有文本时不会制造提示；`no_feedback` 有文本的异常另行统计。原始 `results.json` 仅用于审计反馈标记与文本一致性，不将完整日志混入模型输入。由于日志格式可能附加装饰，逐字找不到反馈文本只构成待核查项，不自动认定 CSV 错误。

`labels.<split>.jsonl` 单独存放 `{schema_version, sample_id, target_timestamp, target_code, target_results}`。不将真实下一代码或下一测试结果放入输入、候选生成、参考匹配或测试时分组选择。标识匹配必须使用 `sample_id`，不能依赖文件顺序碰巧相同。未来反馈不进入当前输入，也不作为预测目标。

## 题目与划分

题目文本来自 `autograders/` 下唯一匹配的作业目录中的全部 `*_desc.txt`，按路径排序拼接。依照上游实际目录命名，先取 `_autograder` 之前的作业名，再去除连字符/下划线并忽略大小写，例如 `lab03_autograder_github_ai → lab03`、`preLab01_autograder_github → pre-lab01`。它是官方作业级描述集合，不是猜测的函数到题目映射，也不含参考实现。每个作业的来源路径和映射状态列于 `audit.json.problem_statements`；同时列出观测源文件、函数名、描述文件标识及没有同名描述的函数，供覆盖审查。仅有 PDF、无文本，或已观测函数名没有相应描述时，保留 `null`；真实模型推理必须拒绝这些样本。实源发现 lab10 带有 lab09 的文件读写描述，却实际考查 thermostat 类，因此必须拒绝这组非空但不适配的文本。文本非空本身不等于覆盖完整，使用前仍须实地验收相应作业的描述覆盖与文件适配。

学生 ID 排序后，以独立 `random.Random(seed)` 打乱，前 `floor(0.8N)` 为 train，随后 `floor(0.1N)` 为 dev，余下为 test。默认种子为 `20260929`。全部观测学生参与划分，包括没有有效修订对的学生，以免根据后续可用性重抽划分。同一学生的所有作业和文件始终在同一划分。`splits.json` 固化完整映射；开发调参只能用 train/dev，最终 test 不参与参考历史拟合、阈值或方法选择。

## 审计和验收边界

`audit.json` 报告 CSV 行数与重复、提交数、学生数、作业数、时间重排、赋予条件与实际下发的差别、测试掩码缺失、代码缺失、同代码修订、多文件范围、题目文本覆盖，以及每划分和每作业的样本量。`source_files` 清单只含本地相对路径与散列，不含原始程序文本。

合成单元测试检查行聚合、重排、历史防泄漏、物理标签隔离、缺代码不跨越、目录存在但 CSV 缺记录、重复代码保留、多文件计数、掩码缺失、学生划分、重建确定性、覆盖保护和符号链接越界防护。真实上游构建和统计结果应由对应运行报告给出，不能用这些合成测试代替。

本模块不导入或执行任何学生代码、第三方测试或反馈生成脚本。可执行测试重放必须单独完成隔离、禁网与资源限制验收，且测试数为零不能算重放通过。

## 2026-09-29 本机实源验收

完整官方归档已获取，归档内 Git 提交为 `e020c7c013187dba7b4130991eadc539d115526a`，压缩文件 SHA-256 为 `b20511be99ce4167a980282e21fab320646c17f05aaea47119306091ecaf8da9`；共 24,050 个文件、32,672,349 个解压字节。`data/prepared/progfeed-v1` 保留为初始静态审计快照，含尚未拒绝的 lab10 错配描述；正式后续使用 `data/prepared/progfeed-v2`。二者不是抽取不同学生的两个实验集，v2 修正题目可用性检查，不改学生划分或修订对。

本机项目 `.venv` 完成 v2 构建，11 项合成数据测试通过；dev/test 均通过推理入口的只读 schema 检查。检查允许题目缺失以验证完整数据，不表示缺题目的记录可以进行真实推理。所读取原始文件清单指纹为 `28790ea549c7544d99568cf29eb7badf70eb82a9a8063fcbeb11aa0a1fd1a4af`。

| 统计 | 数量 |
|---|---:|
| 学生 / 作业 / 学生—作业轨迹 | 215 / 17 / 2,492 |
| CSV 测试行 / 聚合提交 / 可读源文件状态 | 17,385 / 7,130 / 9,490 |
| 有效同名源文件相邻修订对 | 5,355 |
| train / dev / test 修订对 | 4,299 / 513 / 543 |
| train / dev / test 学生 | 172 / 21 / 22 |
| 有实际反馈的修订对 / 代码完全不变的修订对 | 994 / 1,633 |
| 相邻同名文件不存在的排除对 | 462 |
| 没有可观测下一提交的文件状态 | 3,673 |
| 涉及多文件提交的修订对 | 1,897 |
| 当前或目标缺 CSV 测试行的修订对 | 342 |
| 缺逐测试掩码的 CSV 行 | 8,333 |
| 没有对应 CSV 行的提交目录 | 437 |
| 缺少 / 无法解析的 results.json | 1 / 14 |
| 没有可读代码的提交 / 预期代码读取失败 | 28 / 0 |
| 缺可用题目文本的修订对（含 lab10 错配） | 2,764 |

未发现完全重复 CSV 行、非法测试掩码、时间顺序重排或 `no_feedback` 条件却有非空反馈文本。6,525 行被分配到 `tc/nl`，其中 5,099 行没有实际下发内容；实际有反馈的是 1,426 行。不能把分配条件的行数当成反馈下发数量。

日志中另有 93 个提交（lab07 36 个、lab09 57 个）包含反馈标题但 CSV 没有实际文本。已逐条检查：93 个提交都属于 `no_feedback`；其中 110 条测试输出在标题后、`Test Failed:` 前全部只有空白。对应 `feedback_generation.py` 对非 1/2 组返回空串，而 `test.py` 仍打印标题。因此这批是空标题，不是已发现的反馈文本遗漏；`audit.json` 的 marker 计数不应解释为实际收到提示。

### 首轮明确的单文件题目范围

以下 5 组已逐一核对官方 `files.txt`、`test.py`、全部描述正文和 CSV 函数名。这里确认的是题目与源文件对应，未声称判题程序已经安全执行或逐例重放一致。

| 作业 | 唯一预期源文件 | 描述覆盖函数 |
|---|---|---|
| lab02 | `to_do_list.py` | `add_task`, `delete_task`, `move_task` |
| lab03 | `triangle_class.py` | `is_edge_sorted`, `classify_by_edges`, `classify_by_angles` |
| lab06 | `nested_loops.py` | `get_names`, `average_scores` |
| lab07 | `dictionaries.py` | `count_words`, `average_prices`, `count_bigrams` |
| lab09 | `files.py` | `print_stars_to_file`, `calc_avg_from_file` |

lab05 的描述与函数对应，但正式任务需要 3 个源文件，不属于首轮单文件范围。lab10 的文件读写描述与 thermostat 任务不相符，v2 已拒绝；其余 10 个作业没有 `_desc.txt`，后续须验收 PDF 题目提取后再扩展，不能拿空题目跑模型。

仅依据当前可见信息选择上述 `(lab, source_file)`、当前提交目录确实只有该 `.py` 文件，得到 train 1,391、dev 197、test 138 个样本。再限定实际非空反馈，得到 train 737（140 学生）、dev 91（19 学生）、test 48（15 学生）；这些是完整合格范围，未根据下一代码或下一结果优选。若机制分析还要求非空过去历史，对应为 519 / 70 / 31；这项额外范围必须单独报告，空历史不等于数据损坏。参考匹配还可能有单列的可用性限制，不可悄悄改动上述总体。

### 上游判题资源的静态覆盖

17/17 个作业目录都包含 `test.py`、`run_tests.py`、`run_autograder`。各主 `test.py` 合计有 47 个未注释的 `test_*` 方法定义：lab00/01/02/03/05/07、pre-lab01/02/05 各 3 个；lab06/09、pre-lab03/06/07/08 各 2 个；lab10/11 各 4 个。当前检查证明存在非空判题实现，不能等同于执行时确实发现并通过 47 项。

测试中引用的本地 `cull_input.py`、`feedback_generation.py` 等辅助模块存在。lab02/03 另有 `test_openai.py`，它是付费 API 调用示例，不是独立保留评测集；lab06 另有 `test copy.py` 备份。在已发布判题目录中，未找到明确划分的独立 held-out 测试套件。因此今后若使用这些测试为候选分组，不能再称同一测试上的结果为独立测试泛化。

原始 `run_autograder` 带有 `/autograder` 固定路径、修改/复制文件和上传脚本调用，部分作业还做 `cull.py` 源码预处理。17 个目录均未发布 `env.sh`、`upload_to_github.sh` 和 `consent.csv`；这是公开版本剔除部署/个人信息后的资源边界，不能补造真实用户信息。反馈生成模块会导入 API 客户端，并存在缺 consent 文件时继续尝试反馈的路径。必须在后续独立的禁网、受限隔离适配中关闭反馈生成与上传，并记录与原始判题的差异；本轮没有运行这些脚本，也没有进行学生代码执行。

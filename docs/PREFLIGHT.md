# 本地分词器上下文检查

`student_sim_cd.preflight` 在启动模型前，逐条检查全部输入和四个评分条件的上下文长度。它只加载指定本地目录中的分词器，使用 `local_files_only=True`、`trust_remote_code=False` 并开启离线标志；不加载模型权重、不联网、不读取标签、不执行学生代码。分词计算在 CPU 上完成。

在项目根目录运行，例如检查完整 matched dev 集：

```sh
PYTHONPATH=src .venv/bin/python -B -m student_sim_cd.preflight \
  --inputs data/prepared/progfeed-selected-v1/matched/inputs.dev.jsonl \
  --references data/prepared/progfeed-selected-v1/matched/references.dev.jsonl \
  --model-path /absolute/path/to/local-model-tokenizer \
  --output outputs/preflight-matched-dev-v1 \
  --context-limits 8192 16384 32768 \
  --max-new-tokens 512 1024 2048 4096
```

test 集使用对应的 `inputs.test.jsonl`、`references.test.jsonl` 和新的输出目录。分词器文件齐备即可检查，无需等待权重下载完成。需要项目环境安装与正式推理一致版本的 Transformers/tokenizers；本命令不会自动安装或下载依赖。

`--references` 可省略，但此时按正式推理规则使用空参考历史，并明确标记 `empty_reference_diagnostic`，不能把它视为匹配参考主实验。提供参考文件时必须覆盖全部输入 ID，不允许悄悄回退。输入仍经正式推理的字段与历史时间校验；缺题目或混入未来标签字段会拒绝运行。

## 计算规则

检查复用 `inference.build_messages` 和相同的 chat template：四条件为 `11/01/10/00`，分词时 `tokenize=True`、`add_generation_prompt=True`，明确关闭截断。当前完整代码的复制候选按 `encode(code, add_special_tokens=False)` 加恰好一个 EOS 计算；空代码是一个 EOS，不会丢弃。代码内若出现保留特殊 token，会逐条标出与正式评分协议不兼容，不会静默删样本。

记样本 `i` 的最长分支提示为 `P_i`，当前代码加 EOS 长度为 `C_i`，生成预算为 `B`。能够完成当前推理协议的长度条件是：

```text
P_i + max(B, C_i) <= max_context_tokens
```

报告分别给出生成预算、复制候选和二者同时满足的数量，并给出每个分支的生成预算覆盖。全样本需要的最小上下文为 `max_i(P_i + max(B, C_i))`；不会把来自不同样本的最长提示和最长代码直接相加。若本地 `config.json` 提供 `max_position_embeddings`，会同时报告所需长度是否超过该声明，但不会修改配置或自动启用长度扩展。

## 输出与解释

输出目录必须不存在。没有覆盖、续写或自动挑选样本的行为。

| 文件 | 内容 |
|---|---|
| `manifest.json` | 输入、参考文件及分词器文件散列；配置、实现与提示版本；依赖版本和实际分词器类/EOS/模板指纹 |
| `lengths.jsonl` | 每个样本的四分支提示 token 数、复制代码及 EOS 长度、提示加复制长度和特殊 token 检查 |
| `report.json` | 全体最大长度、各预算的最小上下文需求、完整覆盖表及所有溢出样本 ID |

分词器指纹只读取分词器、模板与作为回退信息的 `config.json`，不会遍历或散列 `.safetensors` 等权重。默认比较 8,192 / 16,384 / 32,768 上下文与 512 / 1,024 / 2,048 / 4,096 生成预算；这是一份参数对照报告，不会据此改正式推理配置，不会截断历史、截断提示、删除溢出记录或产生筛选后的数据集。

未知的生成候选在解码后重新分词，其最终规范化评分长度仍需正式推理逐个检查。token 数覆盖不证明显存够用、执行速度可接受或模型方法有效。本模块的合成测试使用假分词器；只有实际运行本地目标分词器后，输出才是该模型的真实长度检查结果。

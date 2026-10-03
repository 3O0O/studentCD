# 复现代码与实验协议

本仓库包含数据重建、文本候选生成、四条件评分、静态评价、训练学生概率预测及独立复核代码。已完成实验的聚合结果见 [研究概要](RESEARCH_SUMMARY.md)。原始学生代码、处理数据、生成正文、模型权重和逐记录结果不随仓库发布；因此克隆仓库能够运行合成测试，不能直接重建历史真实运行的全部产物。

## 环境与合成测试

基础数据处理和大部分测试仅依赖 Python 标准库，声明支持 Python 3.9 及以上。下面的测试不下载数据或模型，不运行学生代码；涉及可选模型依赖的测试会在依赖缺失时明确跳过。

```sh
python3 -m venv .venv
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -t . -v
```

模型推理依赖声明在 [pyproject.toml](../pyproject.toml)，既有服务器环境的版本观察记录在 [环境清单](../configs/environment.server-20260929.json)。该清单不是带 wheel 哈希的依赖锁文件。按自己的平台事先安装依赖，并独立提供兼容的本地模型权重；历史模型为 Qwen2.5-Coder-7B-Instruct。真实推理以 `local_files_only=True`、`trust_remote_code=False` 和 safetensors 加载，不自动下载或安装。模型获取与来源校验工具见 `scripts/download_model.py`；它只应在明确允许联网和下载的环境中单独调用。

## 数据准备

主数据来源是 [ProgFeed 官方公开仓库](https://github.com/umass-ml4ed/progFeed-dataset-public)。数据许可、提交重建、学生划分、单文件范围、实际反馈定义和未来标签隔离详见 [数据协议](DATA_PROTOCOL.md)。使用者需要自行取得合法可用的数据并保留官方归属。仓库不再分发第三方教育数据。

准备完成后，可做只读 schema 验证：

```sh
PYTHONPATH=src .venv/bin/python -B -m student_sim_cd.inference \
  --inputs /path/to/prepared/inputs.dev.jsonl --validate-only
```

`--validate-only` 不加载模型，也不证明数据来源或标签无泄漏。正式输入与下一提交标签须分文件；真实下一代码不能加入候选池、参考匹配或推理输入。完整历史和上下文长度检查见 [推理协议](INFERENCE.md) 与 [运行前检查](PREFLIGHT.md)。

## 通用运行入口

从仓库根目录查看参数：

```sh
.venv/bin/python -B scripts/extract_progfeed.py --help
PYTHONPATH=src .venv/bin/python -B -m student_sim_cd.preflight --help
.venv/bin/python -B scripts/run_experiment.py --help
```

`scripts/run_experiment.py` 串联共享候选生成、base/history_d0/CD/B/copy 选择和静态评价。输入、标签、参考历史、模型、输出新目录及设备均须显式指定。调用方负责真实 GPU 资源许可、线程、进程组、预算、截止时间及日志；命令入口本身不构成安全沙箱。不得把合成测试通过或 `SUCCESS` 当作方法有效性证据。

## 已冻结协议与复核器

| 阶段 | 配置与实现 | 已完成范围 |
| --- | --- | --- |
| 首轮共享候选 | `configs/first_run.json`、`scripts/run_experiment.py` | 文本候选与四条件评分 |
| 缓存规范化重评分 | `configs/cache_rescore_v1.json`、`scripts/run_cached_experiment.py` | 70 dev、17 学生、318 候选 |
| 历史替换干预 | `configs/history_interventions_v1.json`、`scripts/run_history_interventions.py` | 同一 dev、候选和提取协议 |
| 候选支持与提示控制 | `configs/candidate_support_v1.json`、`scripts/run_candidate_support.py` | 1260 新主尝试与 1820 提示尝试 |
| A 静态概率原型 | `configs/a_static_v1.json`、`scripts/run_a_static.py` | 737 训练记录、140 学生与同一 70 dev |

历史协议保留原字节、来源哈希、相对产物位置和当时资源请求状态，不为出版重写成已获批准的通用配置。里面的模型路径、截止时间和缓存哈希是历史运行记录；相应私有输入与缓存不在仓库中。配置中的历史状态不等于 [研究概要](RESEARCH_SUMMARY.md) 所记的最终完成状态，也不构成新的运行授权。

`run_candidate_support.py`、`run_a_static.py` 及 A 正文复核分支带有原实验环境的固定主机、账号、规范化目录和受控进程组检查。这些保护保持原实现，克隆后不能在任意主机直接运行；不应删除检查来绕过限制。通用分析逻辑位于 `src/student_sim_cd/`，移植专用启动流程需要单独评审自己的路径、资源控制和数据边界。实验室 SSH 连接、凭据、机器配置和本机运维助手不属于本仓库。

独立标准库复核器是 `scripts/verify_candidate_support_results.py` 和 `scripts/verify_a_static_results.py`。它们检查指标、归一化、来源及哈希等声明；服务器正文复核与本地无代码数值复核的范围不同。数值 verified 不包含 GPU 或进程释放，资源收尾需另外检查。历史真实产物未公开，不能把一次对合成夹具的复核称为重现真实结果。

## 尚未完成的执行评价

所有学生提交及第三方判题程序都视为不可信代码。本项目尚未验收判题隔离、禁网、资源限制和原判题结果重放，不允许直接执行这些程序。虚拟环境、tmux、工作目录和超时各自都不足以提供隔离。

当前 A 只按文本是否与当前提交完全相同分组。测试通过变化、回归、反馈落实正确性及多步教学效果不是已完成评价。未来执行分组还要处理编译失败、缺测和混合状态变化，并区分用于选择候选的测试与独立评测测试。

## 发布与许可

当前只发布项目代码、合成测试、冻结配置和聚合研究文档；本机原始运行记录保持原样。源码尚未指定发布许可证，公开可见不等于授予开源再使用许可。第三方数据与模型分别遵守原许可。

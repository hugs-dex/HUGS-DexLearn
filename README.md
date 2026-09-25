# HUGS-DexLearn

Human Prior 与机器人抓取模型的研究代码。Python 包名保持 `dexlearn`。
本版本采用 **research-only** 发布说明，首发交付 **source + contract**：源码、配置、接口文档和可运行的接口测试。
训练、采样、导出、评估和可视化代码均保留；这不表示完整训练或端到端 Human Prior 已复现。

## 保留的功能

| 功能 | 源码与配置 | 首发验证边界 |
| --- | --- | --- |
| Shadow、Leap-SP；兼容 Leap | 保留机器人训练、采样、类型评估、可视化 | 配置、入口、数据字段和输出接口；真实训练待验证 |
| Human Prior | 保留 Proposed、Independent、Joint、Reverse、Legacy | 配置、CPU 模型单元测试、五类导出接口；真实权重推理待验证 |
| scene budget | 保留几何标签、budget head、旧标签 baseline | 入口与配置；真实标签生成和训练待验证 |
| 旧 research baseline | 保留全部原有 algo YAML 和网络实现 | 合法配置配对检查；实验效果待验证 |
| 单/多 GPU 编排 | 保留两阶段训练与多 worker 采样 | dry-run、唯一分配、子进程失败传播；真实多 GPU 运行待验证 |
| 结果与 prior 可视化 | 保留 trimesh/Viser 入口 | 参数与模块导入；交互显示及 MANO 资源待验证 |

## 快速检查

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[contract]'
python -m pytest tests/public -q
python -m dexlearn.main task=sample --cfg job --resolve
```

这些检查不需要 GPU、checkpoint、MANO 或数据集。完整 CPU contract 套件需要
[安装说明](docs/installation.md) 中的 runtime 依赖。
不使用开发机的 editable 安装，也不把相邻私有研究仓库加入 `PYTHONPATH`。

## 数据和资产

```bash
# 指向包含 object/、OurHumanGraspFormat/ 的实际 bundle，不是 archives/ 的外层目录。
export ANYSCALEGRASP_DATA_ROOT=/path/to/data-bundle
export HUGS_ASSET_ROOT=/path/to/robot-assets
export MANO_ROOT=/path/to/licensed-mano-models
export HUGS_OUTPUT_ROOT=/path/to/new-run-output
```

`AnyScaleGraspDataset` 是兼容别名；两者同时设置时以 `ANYSCALEGRASP_DATA_ROOT` 为准。
默认资源路径为本仓库 `assets/`，没有隐式研发数据 fallback。
机器人资产根下应有 `robot/shadow_hand/` 和 `robot/leap_hand/`。
MANO 根目录下应有 `MANO_RIGHT.pkl`、`MANO_LEFT.pkl`。
数据、checkpoint、MANO、mesh 归档均不随源码分发，也不自动下载。
本仓库不向 Hugging Face 上传任何内容。

## 工作流入口

以下是依赖和输入齐全后的运行命令，非首发端到端验证声明。checkpoint 必须显式替换为已有模型文件。

```bash
# Human Prior 训练；机器人训练改为 algo=robotMultiHierar data=shadowMulti 或 leapspMulti。
python -m dexlearn.main task=train algo=humanMultiHierar data=humanMulti exp_name=prior
# 采样
python -m dexlearn.main task=sample algo=humanMultiHierar data=humanMulti test_data=humanMulti ckpt=/path/to/model.pth
# BODex 所需 index-MCP prior 导出；独立训练分支分别提供 score/pose checkpoint。
python -m dexlearn.main task=obj_human_prior_export data=humanMulti algo=humanMultiHierar test_data=DGNMulti task.score_ckpt=/path/to/type.pth task.pose_ckpt=/path/to/pose.pth task.output_dir=/path/to/new-prior
# 读取已保存的评分结果，不在 evaluate 中重新采样。
python -m dexlearn.main task=evaluate data=humanMulti algo=humanMultiHierar task.human_results_dir=/path/to/human-scores task.dgn_results_dir=/path/to/object-scores
# 查看已采样结果，或使用 task=visualize_human_prior task.prior_dir=/path/to/robot-prior。
python -m dexlearn.main task=visualize data=shadowMulti test_data=shadowMulti algo=robotMultiHierar ckpt=/path/to/model.pth
# scene budget 保留全部 mode；all 包含标签生成和训练。
python -m dexlearn.main task=scene_budget data=humanMulti algo=humanMultiHierar task.mode=all
```

`human_preprocess` 会写入 grasp 文件，`human_prior_format` 会写出格式化结果；在明确的工作副本上运行。
普通读取默认禁用人手 pose-group cache 写回，输出和日志使用独立目录。
评估的 scale-anchor baseline 需要 `metadata/Ours_object_scale_type_distribution.json`，缺失时显式提供
`task.human_scale_anchor_distribution_json`，不能把缺失 baseline 当作已评测。

- [安装与依赖](docs/installation.md)
- [数据、prior 与 checkpoint contract](docs/contracts.md)
- [完整保留工作流和 baseline 配对](docs/workflows.md)
- [首发验证与已知限制](docs/validation.md)
- [发布与维护](docs/releasing.md)
- [Research-only 说明](RESEARCH_ONLY.md)

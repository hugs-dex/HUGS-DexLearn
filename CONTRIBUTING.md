# 贡献与维护

修改进入功能所属模块，保留 `dexlearn` 包名与已有任务入口。
接口变更必须同时说明 producer、consumer、字段形状、米单位、坐标系、尺度、五类 ID 和兼容方式。

先运行 `python -m pytest tests/public -q`，安装完整 CPU 依赖后运行 `python -m pytest tests -q`。
提交实际命令与结果；未运行的 GPU、MANO、数据或端到端步骤标为 not-run。
不提交数据、权重、缓存、日志、机器路径或凭据。修复用独立 commit 移植，避免目录全量覆盖。

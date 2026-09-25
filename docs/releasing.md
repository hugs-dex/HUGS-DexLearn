# 发布与维护

本仓库使用独立 Git 历史。公开提交只包含源码、配置、测试与说明，不包含私有历史、数据、权重或输出。
私有来源映射与迁移前后文件 hash 存在内部交付记录；公共 release manifest 只记录公开依赖和验证边界。

维护人由项目组织安排，CODEOWNERS 提供候选 review 路由。修复以独立 commit 选择性双向移植，逐项核对文件映射与测试；
不使用私有目录全量覆盖社区修改。HUGS 入口只能在真实集成验证后更新公开组件 lock。

发布前运行 `scripts/audit_source.py`、轻量测试、完整 CPU contract、构建 wheel，并从独立目录安装 wheel 验证配置资源。
检查历史、依赖固定版本和数据缺失说明。不得把未运行的 CI、GPU、可视化或端到端步骤写成通过。
本轮按用户要求不处理许可证；保持 research-only 说明，不以许可证工作阻塞技术验收。

GitHub 标准公开源码仓库和标准公开 Actions runner 的计划费用为 $0；larger runner/GPU/数据传输另计。
不使用 Git LFS，也不在 Hugging Face 建仓库或上传内容。
发布远端候选、PR 或正式 public 切换由项目负责人明确确认后执行。

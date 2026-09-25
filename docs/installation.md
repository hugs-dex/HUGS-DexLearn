# 安装与依赖

## 配置与轻量 contract

使用 Python 3.10 创建独立环境，安装 `python -m pip install -e '.[contract]'`。
`python -m pytest tests/public -q` 不需要 torch、MANO、MinkowskiEngine 或真实数据。
Hydra 配置随 wheel 打包，`dexlearn --cfg job --resolve` 可在安装后的任意目录运行。

## CPU 模型和导出 contract

本次验证组合：Python 3.10、PyTorch 2.2.2（CUDA 12.1 wheel，但验证时隐藏 GPU）、NumPy 1.26.4、
SciPy 1.14.1、PyTorch3D 0.7.8、Hydra 1.3.2。测试不构建 MinkowskiEngine，不调用 GPU。

```bash
python -m pip install -c releases/cpu-contract-constraints.txt -e '.[runtime,contract]'
git submodule update --init third_party/nflows third_party/diffusers
python -m pip install ./third_party/nflows ./third_party/diffusers 'huggingface-hub<1' iopath fvcore
```

PyTorch3D 的原始兼容组合是 Python 3.10 / PyTorch 2.2.2 / CUDA 12.1 / PyTorch3D 0.7.8。
官方 conda 包入口：
`https://api.anaconda.org/download/pytorch3d/pytorch3d/0.7.8/linux-64/pytorch3d-0.7.8-py310_cu121_pyt222.tar.bz2`。
可用 conda 在相同版本组合中安装，或对本次 CPU contract 的独立 venv 使用提供的辅助脚本：

```bash
python scripts/install_pytorch3d_contract.py --archive /path/to/pytorch3d-0.7.8-py310_cu121_pyt222.tar.bz2
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python -m pytest tests -q
```

辅助脚本只把经 SHA-256 校验的包文件解包到当前 venv，不修改系统/其他环境；它不是通用 conda 安装器。
本次实测覆盖 transforms 和模型/导出 CPU contract，不声称其 CUDA/C++ 算子完整可用。
完整 GPU workflow 仍应使用官方安装方式并实际构建、检查扩展。

## 训练、采样与可视化 runtime

六个外部依赖的公开 HTTPS URL 与固定 commit 见 `.gitmodules` 和 `releases/dependencies.json`。
可以按需 `git submodule update --init third_party/<name>`，不要要求仅检查配置的用户初始化全部依赖。

| 依赖 | 使用位置 | 安装与验证边界 |
| --- | --- | --- |
| MinkowskiEngine | 稀疏点云网络与 collate | 固定 commit；编译前检查 PyTorch/CUDA、C++、OpenBLAS；首发未验证 GPU 构建 |
| nflows、diffusers | flow、扩散与 type embedding | 固定 commit；CPU contract 已验证；不需要 Hub 模型下载 |
| PyTorch3D | 旋转变换及人手计算 | 上述 0.7.8 组合；CPU transforms 已验证 |
| manopth | 人手预处理、格式化和可视化 | 固定 commit，自行提供 MANO；按 task 延迟加载 |
| pytorch_kinematics、utils_python | 机器人可视化 | 固定 commit，`pip install ./third_party/<name>`；交互显示未验证 |
| Viser/trimesh/OpenCV | 查看与渲染 | `pip install -e '.[visualize]'`；端口由 task 配置指定 |

MinkowskiEngine 构建遵循其固定版本 README，根据本机选择 CUDA toolkit，不写死 toolkit 路径，不假设有 sudo。
安装后至少验证 `import MinkowskiEngine` 与小型稀疏网络；未完成前不要声称训练已就绪。
本仓库的 `task=... --cfg job --resolve` 仅检查配置，不会启动 GPU、服务器或训练。

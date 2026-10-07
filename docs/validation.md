# 首发验证与限制

首发范围是 source + contract；所有指定功能源码保留。测试使用独立 Python 3.10 环境，
不借用私有研发 checkout 的 editable 安装或模型输出。

| 检查 | 已运行结果 |
| --- | --- |
| 所有 task 的 Hydra CLI、全部 algo/data/test_data/task YAML 及合法配对 | 通过；旧 humanMultiDiffusion 的两个未使用插值已移除 |
| 配置根目录优先级、缺失数据/MANO/checkpoint 诊断 | 通过 |
| 多 worker 唯一分配、真实失败子进程、shell 返回码、dry-run 无日志写入 | 通过 |
| 主 Human Prior、Joint/Reverse 模型与导出、selection metadata、scene whitelist | CPU 测试通过 |
| 五类 export shape、wxyz、非有限数、checkpoint hash 和 manifest | CPU 测试通过 |
| 11 个 task 模块导入 | 通过；无需导入 MANO 或 MinkowskiEngine |
| 独立 HUGS-BODex reader | 合成 prior 从 exporter 保存，再由真实 consumer 读取并验证，通过 |
| Public dataset | DGNMulti 和 humanMulti 各一个 test item 读取通过；未做全量审计 |

复查命令（从本仓库运行）：

```bash
python scripts/audit_source.py
python -m pytest tests/public -q
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests -q
python -m compileall -q dexlearn tests scripts
python -m build
CUDA_VISIBLE_DEVICES= python scripts/check_bodex_contract.py --bodex-root /path/to/HUGS-BODex
CUDA_VISIBLE_DEVICES= python scripts/check_public_data.py --bundle /path/to/data-bundle
```

合成 prior 测试不调用网络推理，不证明模型质量或真实 human synthesis 成功。
真实训练、GPU 扩展构建、真实多 GPU、checkpoint 采样/导出、MANO forward、交互可视化、Benchmark 均未运行。
GitHub CI 文件已提供，未声称远端 Actions 已执行。

已检查的 public bundle 缺少 checkpoint、MANO、导出的 human prior 和部分人手 canonical mesh。
源码保留相应能力；缺少输入时不能以原研发目录作为 fallback。

## Source installation check (2026-09-25)

The updated README installation was tested in a fresh Python 3.10.12 environment
with uv 0.11.32, using only dependencies and build settings in `pyproject.toml`.
PyTorch3D 0.7.8 was built from official commit
`75ebeeaea0908c5527e7b1e305fbc7681382db47`; no conda archive was used.

- All 151 CPU tests passed; a separate installation without submodules passed all 87 public tests.
- PyTorch3D CUDA KNN and MinkowskiEngine CUDA convolution passed forward/backward checks.
- The HUGS point-cloud backbone passed a CUDA forward check; visualization modules imported successfully.
- Dependency checks, Hydra configuration resolution, wheel and source-package builds passed.
- Build platform: Linux x86_64, GCC 11.4, CUDA toolkit 12.4, RTX 4090, PyTorch 2.2.2+cu121.

These extension checks extend the initial validation above. Full training,
checkpoint inference, MANO model execution, and interactive rendering remain untested.

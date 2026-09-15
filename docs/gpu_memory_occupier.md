# GPU 显存占用脚本

`scripts/occupy_gpu_memory.py` 会在 GPU 0、1、2、3 上分别分配并持续持有
20 GiB 显存。按 `Ctrl+C` 结束程序后，CUDA 会自动释放显存。

在 `spatial` 环境中运行：

```bash
python scripts/occupy_gpu_memory.py
```

可另开终端查看占用情况：

```bash
nvidia-smi
```

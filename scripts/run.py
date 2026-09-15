"""在 GPU 0、1、2、3 上分别持续占用 20 GiB 显存。"""

import time

import torch


GPU_IDS = (0, 1, 2, 3)
BYTES_PER_GPU = 20 * 1024**3


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 PyTorch 和 NVIDIA 驱动。")

    device_count = torch.cuda.device_count()
    if device_count <= max(GPU_IDS):
        raise RuntimeError(
            f"当前只能看到 {device_count} 张 GPU，无法使用 GPU {GPU_IDS}。"
        )

    buffers = []
    for gpu_id in GPU_IDS:
        device = torch.device(f"cuda:{gpu_id}")
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < BYTES_PER_GPU:
            raise RuntimeError(
                f"GPU {gpu_id} 可用显存仅 {free_bytes / 1024**3:.2f} GiB，"
                "不足 20 GiB。"
            )

        # uint8 的元素大小为 1 字节，因此张量大小正好为 20 GiB。
        buffer = torch.empty(BYTES_PER_GPU, dtype=torch.uint8, device=device)
        buffer.zero_()
        torch.cuda.synchronize(device)
        buffers.append(buffer)
        # print(f"GPU {gpu_id}: 已占用 20 GiB 显存", flush=True)

    # print("显存占用已保持；按 Ctrl+C 退出并释放显存。", flush=True)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n正在退出，显存将由 CUDA 自动释放。", flush=True)


if __name__ == "__main__":
    main()

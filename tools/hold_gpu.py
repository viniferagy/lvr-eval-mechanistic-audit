#!/usr/bin/env python3
"""Hog all visible GPUs with large matmuls and auto-expanding ballast."""
import os, sys, time
import torch


MB = 1024 * 1024


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _alloc_ballast_chunk(dev, dtype, chunk_mb: int):
    element_size = torch.tensor([], dtype=dtype).element_size()
    numel = max(1, chunk_mb * MB // element_size)
    return torch.empty(numel, device=dev, dtype=dtype)


def expand_ballast(dev, tensors: list, dtype, *,
                   chunk_mb: int,
                   min_chunk_mb: int,
                   reserve_mb: int) -> tuple[int, int]:
    """Allocate ballast until free memory drops near reserve_mb."""
    made = 0
    made_mb = 0
    while True:
        try:
            free_bytes, _ = torch.cuda.mem_get_info(dev)
        except RuntimeError:
            return made, made_mb

        available_mb = max(0, free_bytes // MB - reserve_mb)
        if available_mb < min_chunk_mb:
            return made, made_mb

        attempt_mb = min(chunk_mb, int(available_mb))
        while attempt_mb >= min_chunk_mb:
            try:
                tensors.append(_alloc_ballast_chunk(dev, dtype, attempt_mb))
                made += 1
                made_mb += attempt_mb
                break
            except RuntimeError:
                torch.cuda.empty_cache()
                attempt_mb //= 2
        else:
            return made, made_mb


def main():
    if not torch.cuda.is_available():
        print("[hold_gpu] CUDA not available", flush=True); sys.exit(1)

    n = torch.cuda.device_count()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    print(f"[hold_gpu] PID={os.getpid()} holding {n} GPU(s), "
          f"CUDA_VISIBLE_DEVICES={visible}", flush=True)

    base_size = int(os.environ.get('HOLD_MATRIX_SIZE', '16384'))
    expand_interval = max(1, _env_int('HOLD_EXPAND_INTERVAL_SEC', 10))
    chunk_mb = max(1, _env_int('HOLD_EXPAND_CHUNK_MB', 512))
    min_chunk_mb = max(1, _env_int('HOLD_EXPAND_MIN_CHUNK_MB', 64))
    reserve_mb = max(0, _env_int('HOLD_RESERVE_MB', 512))
    dtype = torch.float32

    triples = []
    ballast_by_gpu = {}
    for i in range(n):
        dev = torch.device(f'cuda:{i}')
        ballast_by_gpu[i] = []
        # 1) 分配 a, b, c（c 是输出 buffer，避免循环中再分配）
        size = base_size
        while size >= 1024:
            try:
                a = torch.randn(size, size, device=dev, dtype=dtype)
                b = torch.randn(size, size, device=dev, dtype=dtype)
                c = torch.empty(size, size, device=dev, dtype=dtype)
                torch.matmul(a, b, out=c)            # 预热，确保 cuBLAS workspace 也分配好
                torch.cuda.synchronize(dev)
                triples.append((i, a, b, c, dev))
                print(f"[hold_gpu] GPU{i}: matmul {size}x{size} (a,b,c ready)",
                      flush=True)
                break
            except RuntimeError:
                a = b = c = None
                torch.cuda.empty_cache()
                size //= 2
        else:
            print(f"[hold_gpu] GPU{i}: failed to allocate matmul tensors",
                  flush=True)
            continue

        # 2) 初始 ballast；后续循环会自动补吃新释放的显存。
        made, made_mb = expand_ballast(
            dev, ballast_by_gpu[i], dtype,
            chunk_mb=chunk_mb, min_chunk_mb=min_chunk_mb, reserve_mb=reserve_mb,
        )
        print(f"[hold_gpu] GPU{i}: initial ballast +{made_mb} MiB "
              f"({made} tensors, reserve={reserve_mb} MiB)", flush=True)

    total_tensors = sum(len(v) for v in ballast_by_gpu.values())
    print(f"[hold_gpu] ballast tensors: {total_tensors}; "
          f"auto-expand every {expand_interval}s; entering matmul loop", flush=True)
    next_expand = time.monotonic() + expand_interval
    while True:
        for _, a, b, c, _ in triples:
            torch.matmul(a, b, out=c)
        for _, _, _, _, dev in triples:
            torch.cuda.synchronize(dev)
        if time.monotonic() >= next_expand:
            for i, _, _, _, dev in triples:
                made, made_mb = expand_ballast(
                    dev, ballast_by_gpu[i], dtype,
                    chunk_mb=chunk_mb,
                    min_chunk_mb=min_chunk_mb,
                    reserve_mb=reserve_mb,
                )
                if made:
                    print(f"[hold_gpu] GPU{i}: auto-expanded +{made_mb} MiB "
                          f"({made} tensors, total={len(ballast_by_gpu[i])})",
                          flush=True)
            next_expand = time.monotonic() + expand_interval

if __name__ == '__main__':
    main()

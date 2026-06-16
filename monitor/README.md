# Teutonic Resource Monitor

Small FastAPI dashboard for watching CPU, RAM, disk, GPU, and top processes while
running Teutonic mining/training jobs.

Run locally:

```bash
run/run_monitor.sh
```

Then open `http://127.0.0.1:17888` from an SSH tunnel or local browser in the
same environment.

Config:

```bash
MONITOR_HOST=127.0.0.1 MONITOR_PORT=17888 run/run_monitor.sh
```

Cleanup actions are intentionally guarded:

- `Trim Monitor` runs Python GC, `malloc_trim`, and `torch.cuda.empty_cache()` in
  the monitor process only. It cannot free memory owned by a separate training
  process.
- `Drop Caches` asks the kernel to drop filesystem page cache. It may be blocked
  inside containers.
- `SIGTERM` sends a termination signal to the selected process only after typing
  the exact confirmation prompt.

To free GPU memory held by a stuck training process, terminate that process from
the process table. There is no safe way for this monitor process to clear another
process's PyTorch CUDA allocator without stopping that process.

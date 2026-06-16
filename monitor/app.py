#!/usr/bin/env python3
"""Small resource monitor and guarded cleanup app for Teutonic VM/container runs."""
from __future__ import annotations

import ctypes
import gc
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse


APP_TITLE = "Teutonic Resource Monitor"
DEFAULT_TOP_N = 20
MONITOR_PID = os.getpid()

app = FastAPI(title=APP_TITLE)


def bytes_from_kib(value: str) -> int:
    parts = value.split()
    return int(parts[0]) * 1024 if parts else 0


def read_meminfo() -> dict[str, int]:
    data: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        data[key] = bytes_from_kib(value)
    total = data.get("MemTotal", 0)
    available = data.get("MemAvailable", 0)
    used = max(total - available, 0)
    return {
        "total": total,
        "available": available,
        "used": used,
        "used_percent": (used / total * 100.0) if total else 0.0,
        "swap_total": data.get("SwapTotal", 0),
        "swap_free": data.get("SwapFree", 0),
        "cached": data.get("Cached", 0),
        "buffers": data.get("Buffers", 0),
    }


def disk_usage(path: str) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": path,
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "used_percent": (usage.used / usage.total * 100.0) if usage.total else 0.0,
    }


def read_load() -> dict[str, Any]:
    load1, load5, load15 = os.getloadavg()
    return {
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "cpu_count": os.cpu_count() or 0,
        "uptime_seconds": float(Path("/proc/uptime").read_text().split()[0]),
    }


def run_command(args: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def query_gpus() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    query = (
        "index,name,uuid,driver_version,pstate,clocks.gr,clocks.mem,clocks.sm,"
        "clocks.max.gr,clocks.max.mem,memory.total,memory.used,memory.free,"
        "utilization.gpu,utilization.memory,temperature.gpu,temperature.memory,"
        "power.draw,power.limit,fan.speed,pcie.link.gen.current,pcie.link.width.current"
    )
    proc = run_command([
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ])
    if proc.returncode != 0:
        return [{"error": proc.stderr.strip() or proc.stdout.strip()}]
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 22:
            continue
        memory_total_mib = int(float(parts[10]))
        memory_used_mib = int(float(parts[11]))
        memory_free_mib = int(float(parts[12]))
        rows.append({
            "index": int(parts[0]),
            "name": parts[1],
            "uuid": parts[2],
            "driver_version": parts[3],
            "pstate": parts[4],
            "clock_graphics_mhz": none_or_int(parts[5]),
            "clock_memory_mhz": none_or_int(parts[6]),
            "clock_sm_mhz": none_or_int(parts[7]),
            "clock_graphics_max_mhz": none_or_int(parts[8]),
            "clock_memory_max_mhz": none_or_int(parts[9]),
            "memory_total_mib": memory_total_mib,
            "memory_used_mib": memory_used_mib,
            "memory_free_mib": memory_free_mib,
            "memory_used_percent": (memory_used_mib / memory_total_mib * 100.0)
            if memory_total_mib else 0.0,
            "utilization_gpu_percent": none_or_int(parts[13]),
            "utilization_memory_percent": none_or_int(parts[14]),
            "temperature_c": none_or_int(parts[15]),
            "temperature_memory_c": none_or_int(parts[16]),
            "power_draw_w": none_or_float(parts[17]),
            "power_limit_w": none_or_float(parts[18]),
            "fan_speed_percent": none_or_int(parts[19]),
            "pcie_link_gen_current": none_or_int(parts[20]),
            "pcie_link_width_current": none_or_int(parts[21]),
        })
    return rows


def none_or_float(value: str) -> float | None:
    return None if value in {"[Not Supported]", "N/A", ""} else float(value)


def none_or_int(value: str) -> int | None:
    parsed = none_or_float(value)
    return None if parsed is None else int(parsed)


def query_gpu_processes() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    query = "gpu_uuid,pid,process_name,used_memory"
    proc = run_command([
        "nvidia-smi",
        f"--query-compute-apps={query}",
        "--format=csv,noheader,nounits",
    ])
    if proc.returncode != 0:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append({
            "gpu_uuid": parts[0],
            "pid": int(parts[1]),
            "name": parts[2],
            "gpu_memory_mib": int(float(parts[3])),
        })
    return rows


def proc_status(pid_dir: Path) -> dict[str, str]:
    status_path = pid_dir / "status"
    if not status_path.exists():
        return {}
    data = {}
    for line in status_path.read_text(errors="replace").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            data[key] = value.strip()
    return data


def proc_cmdline(pid_dir: Path) -> str:
    raw = (pid_dir / "cmdline").read_bytes()
    text = raw.replace(b"\0", b" ").decode(errors="replace").strip()
    return text[:300]


def top_processes(limit: int = DEFAULT_TOP_N) -> list[dict[str, Any]]:
    rows = []
    gpu_by_pid = {row["pid"]: row for row in query_gpu_processes()}
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            status = proc_status(pid_dir)
            if not status:
                continue
            pid = int(pid_dir.name)
            rss_kib = int(status.get("VmRSS", "0 kB").split()[0])
            rows.append({
                "pid": pid,
                "name": status.get("Name", ""),
                "state": status.get("State", ""),
                "rss_mib": rss_kib / 1024.0,
                "gpu_memory_mib": gpu_by_pid.get(pid, {}).get("gpu_memory_mib", 0),
                "cmdline": proc_cmdline(pid_dir),
            })
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    rows.sort(key=lambda row: (row["gpu_memory_mib"], row["rss_mib"]), reverse=True)
    return rows[:limit]


def status_snapshot() -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "host": os.uname().nodename,
        "pid": MONITOR_PID,
        "load": read_load(),
        "memory": read_meminfo(),
        "disk": [disk_usage(path) for path in ("/", "/workspace", "/tmp") if Path(path).exists()],
        "gpus": query_gpus(),
        "gpu_processes": query_gpu_processes(),
        "top_processes": top_processes(),
        "actions": {
            "trim_self": "Run gc.collect(), malloc_trim(), and torch.cuda.empty_cache() in the monitor process only.",
            "drop_caches": "Ask the kernel to drop page cache; may be blocked inside containers.",
            "terminate_process": "Send SIGTERM to a selected PID after confirmation.",
        },
    }


def trim_self() -> dict[str, Any]:
    gc.collect()
    malloc_trimmed = False
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trimmed = bool(libc.malloc_trim(0))
    except Exception:
        malloc_trimmed = False
    torch_cache_cleared = False
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch_cache_cleared = True
    except Exception:
        torch_cache_cleared = False
    return {
        "gc_collected": True,
        "malloc_trimmed": malloc_trimmed,
        "torch_cuda_cache_cleared": torch_cache_cleared,
        "note": "This only frees memory owned by the monitor process, not training jobs.",
    }


def drop_caches() -> dict[str, Any]:
    run_command(["sync"], timeout=30.0)
    try:
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
    except OSError as exc:
        raise HTTPException(
            status_code=403,
            detail=f"drop_caches is not permitted in this container: {exc}",
        ) from exc
    return {"dropped": True, "note": "Kernel page cache drop requested."}


def terminate_process(pid: int, sig: str, confirm: str) -> dict[str, Any]:
    if pid <= 1 or pid == MONITOR_PID:
        raise HTTPException(status_code=400, detail="refusing to terminate this PID")
    sig = sig.upper()
    expected = f"{sig} {pid}" if sig == "KILL" else f"TERMINATE {pid}"
    if confirm != expected:
        raise HTTPException(status_code=400, detail=f"confirmation must be exactly: {expected}")
    signum = signal.SIGKILL if sig == "KILL" else signal.SIGTERM
    try:
        os.kill(pid, signum)
    except ProcessLookupError as exc:
        raise HTTPException(status_code=404, detail=f"PID {pid} does not exist") from exc
    return {"pid": pid, "signal": signum.name, "sent": True}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/status")
def api_status() -> JSONResponse:
    return JSONResponse(status_snapshot())


@app.post("/api/actions")
async def api_actions(request: Request) -> JSONResponse:
    payload = await request.json()
    action = payload.get("action")
    if action == "trim_self":
        result = trim_self()
    elif action == "drop_caches":
        if payload.get("confirm") != "DROP_CACHES":
            raise HTTPException(status_code=400, detail="confirmation must be exactly: DROP_CACHES")
        result = drop_caches()
    elif action == "terminate_process":
        result = terminate_process(
            int(payload.get("pid", 0)),
            str(payload.get("signal", "TERM")),
            str(payload.get("confirm", "")),
        )
    else:
        raise HTTPException(status_code=400, detail=f"unknown action: {action}")
    return JSONResponse({"ok": True, "result": result, "status": status_snapshot()})


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Teutonic Resource Monitor</title>
  <style>
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f172a; color: #e2e8f0; }
    header { padding: 20px 24px; background: #111827; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
    main { padding: 20px 24px; display: grid; gap: 18px; }
    .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
    .card { background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 16px; box-shadow: 0 8px 20px rgba(0,0,0,.18); }
    .title { color: #93c5fd; font-weight: 700; margin-bottom: 10px; }
    .big { font-size: 28px; font-weight: 800; }
    .muted { color: #94a3b8; font-size: 13px; }
    .bar { height: 10px; border-radius: 99px; background: #334155; overflow: hidden; margin-top: 8px; }
    .fill { height: 100%; background: linear-gradient(90deg, #22c55e, #eab308, #ef4444); }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid #334155; padding: 8px; text-align: left; vertical-align: top; }
    th { color: #93c5fd; }
    .metrics { display: grid; gap: 6px; grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 10px; }
    .metric { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 8px; }
    .metric span { display: block; color: #94a3b8; font-size: 12px; }
    .metric b { display: block; margin-top: 2px; }
    code { color: #fde68a; white-space: pre-wrap; }
    button { background: #2563eb; color: white; border: 0; border-radius: 8px; padding: 8px 12px; cursor: pointer; margin-right: 8px; }
    button.danger { background: #dc2626; }
    button.warn { background: #ca8a04; }
    #message { color: #a7f3d0; }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="big">Teutonic Resource Monitor</div>
      <div class="muted" id="subtitle">Loading...</div>
    </div>
    <div>
      <button onclick="refresh()">Refresh</button>
      <button class="warn" onclick="trimSelf()">Trim Monitor</button>
      <button class="danger" onclick="dropCaches()">Drop Caches</button>
    </div>
  </header>
  <main>
    <div id="message"></div>
    <section class="grid" id="cards"></section>
    <section class="card">
      <div class="title">GPU Processes</div>
      <table>
        <thead><tr><th>GPU UUID</th><th>PID</th><th>Name</th><th>GPU MiB</th></tr></thead>
        <tbody id="gpuProcesses"></tbody>
      </table>
    </section>
    <section class="card">
      <div class="title">Top Processes</div>
      <table>
        <thead><tr><th>PID</th><th>Name</th><th>RSS MiB</th><th>GPU MiB</th><th>Command</th><th>Action</th></tr></thead>
        <tbody id="processes"></tbody>
      </table>
    </section>
    <section class="card">
      <div class="title">Raw Status</div>
      <pre><code id="raw"></code></pre>
    </section>
  </main>
  <script>
    const fmtBytes = b => {
      const units = ["B","KiB","MiB","GiB","TiB"];
      let n = b, i = 0;
      while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
      return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
    };
    const pct = n => `${Number(n).toFixed(1)}%`;
    const val = (v, suffix = "") => (v === null || v === undefined) ? "N/A" : `${v}${suffix}`;
    const bar = n => `<div class="bar"><div class="fill" style="width:${Math.max(0, Math.min(100, n))}%"></div></div>`;
    const card = (title, body) => `<div class="card"><div class="title">${title}</div>${body}</div>`;
    const metric = (name, value) => `<div class="metric"><span>${name}</span><b>${value}</b></div>`;

    async function refresh() {
      const res = await fetch("/api/status");
      const data = await res.json();
      document.getElementById("subtitle").textContent = `${data.host} | PID ${data.pid} | ${new Date(data.timestamp * 1000).toLocaleString()}`;
      const mem = data.memory;
      const cards = [
        card("CPU Load", `<div class="big">${data.load.load1.toFixed(2)}</div><div class="muted">5m ${data.load.load5.toFixed(2)} | CPUs ${data.load.cpu_count}</div>`),
        card("RAM", `<div class="big">${pct(mem.used_percent)}</div><div>${fmtBytes(mem.used)} / ${fmtBytes(mem.total)}</div>${bar(mem.used_percent)}<div class="muted">available ${fmtBytes(mem.available)} | cache ${fmtBytes(mem.cached)}</div>`),
      ];
      for (const gpu of data.gpus) {
        if (gpu.error) {
          cards.push(card("GPU", `<div>${gpu.error}</div>`));
          continue;
        }
        cards.push(card(`GPU ${gpu.index}`, `
          <div class="big">${pct(gpu.utilization_gpu_percent ?? 0)}</div>
          <div>${gpu.name}</div>
          <div class="muted">${gpu.uuid}</div>
          <div>VRAM ${gpu.memory_used_mib} / ${gpu.memory_total_mib} MiB (${pct(gpu.memory_used_percent)})</div>
          ${bar(gpu.memory_used_percent)}
          <div class="metrics">
            ${metric("Memory util", pct(gpu.utilization_memory_percent ?? 0))}
            ${metric("Power", `${val(gpu.power_draw_w, " W")} / ${val(gpu.power_limit_w, " W")}`)}
            ${metric("Temp", `${val(gpu.temperature_c, " C")} / mem ${val(gpu.temperature_memory_c, " C")}`)}
            ${metric("Fan", val(gpu.fan_speed_percent, "%"))}
            ${metric("P-state", gpu.pstate)}
            ${metric("Driver", gpu.driver_version)}
            ${metric("Graphics clock", `${val(gpu.clock_graphics_mhz, " MHz")} / ${val(gpu.clock_graphics_max_mhz, " MHz")}`)}
            ${metric("SM clock", val(gpu.clock_sm_mhz, " MHz"))}
            ${metric("Memory clock", `${val(gpu.clock_memory_mhz, " MHz")} / ${val(gpu.clock_memory_max_mhz, " MHz")}`)}
            ${metric("PCIe", `Gen ${val(gpu.pcie_link_gen_current)} x${val(gpu.pcie_link_width_current)}`)}
          </div>
        `));
      }
      for (const disk of data.disk) {
        cards.push(card(`Disk ${disk.path}`, `<div class="big">${pct(disk.used_percent)}</div><div>${fmtBytes(disk.used)} / ${fmtBytes(disk.total)}</div>${bar(disk.used_percent)}<div class="muted">free ${fmtBytes(disk.free)}</div>`));
      }
      document.getElementById("cards").innerHTML = cards.join("");
      document.getElementById("gpuProcesses").innerHTML = data.gpu_processes.map(p => `
        <tr>
          <td><code>${p.gpu_uuid}</code></td>
          <td>${p.pid}</td>
          <td>${p.name}</td>
          <td>${p.gpu_memory_mib}</td>
        </tr>`).join("") || `<tr><td colspan="4" class="muted">No active compute processes</td></tr>`;
      document.getElementById("processes").innerHTML = data.top_processes.map(p => `
        <tr>
          <td>${p.pid}</td><td>${p.name}<br><span class="muted">${p.state}</span></td>
          <td>${p.rss_mib.toFixed(1)}</td><td>${p.gpu_memory_mib}</td>
          <td><code>${p.cmdline || ""}</code></td>
          <td><button class="danger" onclick="terminatePid(${p.pid})">SIGTERM</button></td>
        </tr>`).join("");
      document.getElementById("raw").textContent = JSON.stringify(data, null, 2);
    }

    async function postAction(payload) {
      const res = await fetch("/api/actions", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
      document.getElementById("message").textContent = JSON.stringify(data.result);
      await refresh();
    }
    async function trimSelf() {
      await postAction({action: "trim_self"});
    }
    async function dropCaches() {
      if (prompt("Type DROP_CACHES to request kernel page-cache drop") !== "DROP_CACHES") return;
      await postAction({action: "drop_caches", confirm: "DROP_CACHES"});
    }
    async function terminatePid(pid) {
      const confirm = prompt(`Type TERMINATE ${pid} to send SIGTERM`);
      if (!confirm) return;
      await postAction({action: "terminate_process", pid, signal: "TERM", confirm});
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("MONITOR_HOST", "127.0.0.1")
    port = int(os.environ.get("MONITOR_PORT", "17888"))
    uvicorn.run("monitor.app:app", host=host, port=port, reload=False)

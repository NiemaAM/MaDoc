"""Resources and durations: what the Monitoring page draws and what /metrics gives to Prometheus.

A thread takes one sample every INTERVAL seconds (CPU, memory, GPU, queue) and keeps the last hour in memory;
jobs.py adds the duration of every step. Nothing is written to disk and nothing of the documents is kept here:
a step is a name, a duration, a device and a number of lines."""

import os
import shutil
import subprocess
import threading
import time
from collections import deque

from madoc_web import config

INTERVAL = 5                       # seconds between two samples
KEPT = 3600 // INTERVAL            # one hour
samples = deque(maxlen=KEPT)
steps = deque(maxlen=300)          # the last steps: {"at", "step", "seconds", "ok", "device", "lines"}
totals = {}                        # (step, ok) -> [count, seconds]: since the start, for /metrics
started = time.time()
_thread = []
_gpu_tool = [shutil.which("nvidia-smi")]
queue_length = [lambda: 0]         # jobs.py gives the function


def gpus():
    """[{"index", "name", "util", "memory_used", "memory_total", "temperature", "power"}] (nvidia-smi), or []."""
    if not _gpu_tool[0]:
        return []
    try:
        out = subprocess.run(
            [_gpu_tool[0], "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        _gpu_tool[0] = None        # no driver here: not asked again
        return []

    def number(value):
        try:
            return float(value)
        except ValueError:
            return None            # "[N/A]"
    found = []
    for row in out.strip().splitlines():
        cells = [cell.strip() for cell in row.split(",")]
        if len(cells) == 7:
            found.append({"index": int(cells[0]), "name": cells[1], "util": number(cells[2]),
                          "memory_used": number(cells[3]), "memory_total": number(cells[4]),
                          "temperature": number(cells[5]), "power": number(cells[6])})
    return found


def sample():
    """One sample of the machine and of this process."""
    point = {"at": time.time(), "cpu": None, "memory_used": None, "memory_total": None, "process_memory": None,
             "disk_used": None, "disk_total": None, "queue": queue_length[0](), "gpus": gpus()}
    try:
        import psutil
        memory = psutil.virtual_memory()
        point.update(cpu=psutil.cpu_percent(interval=None), memory_used=memory.total - memory.available,
                     memory_total=memory.total, process_memory=psutil.Process().memory_info().rss)
    except ImportError:
        pass
    try:
        usage = shutil.disk_usage(config.DATA_DIR if config.DATA_DIR.is_dir() else config.DATA_DIR.parent)
        point.update(disk_used=usage.used, disk_total=usage.total)
    except OSError:
        pass
    return point


def _run():
    while True:
        try:
            samples.append(sample())
        except Exception:      # a sample that fails must not stop the next ones
            pass
        time.sleep(INTERVAL)


def start():
    if not _thread:
        _thread.append(threading.Thread(target=_run, daemon=True, name="monitor"))
        _thread[0].start()


def record_step(step, seconds, ok, device, lines=None):
    steps.append({"at": time.time(), "step": step, "seconds": round(seconds, 3), "ok": ok, "device": device,
                  "lines": lines})
    total = totals.setdefault((step, ok), [0, 0.0])
    total[0] += 1
    total[1] += seconds


def loaded_models():
    try:
        import kraken_engine
        return kraken_engine.loaded()
    except Exception:
        return []


def snapshot(minutes=15):
    """What the Monitoring page asks: the samples of the last minutes, the last steps and their summary."""
    since = time.time() - minutes * 60
    points = [point for point in list(samples) if point["at"] >= since]
    summary = {}
    for step in list(steps):
        entry = summary.setdefault(step["step"], {"count": 0, "errors": 0, "seconds": []})
        entry["count"] += 1
        entry["errors"] += 0 if step["ok"] else 1
        if step["ok"]:
            entry["seconds"].append(step["seconds"])
    for entry in summary.values():
        values = sorted(entry.pop("seconds"))
        entry["mean"] = round(sum(values) / len(values), 2) if values else None
        entry["p95"] = values[min(len(values) - 1, int(len(values) * 0.95))] if values else None
        entry["last"] = None
    for step in list(steps):
        if step["ok"]:
            summary[step["step"]]["last"] = step["seconds"]
    return {"now": time.time(), "interval": INTERVAL, "uptime": time.time() - started, "samples": points,
            "steps": list(steps)[-50:][::-1], "summary": summary, "models": loaded_models(),
            "device": config.device(), "cpu_count": os.cpu_count() or 1}


def prometheus():
    """The same numbers in the text format of Prometheus (GET /metrics)."""
    point = samples[-1] if samples else sample()
    lines = ["# TYPE madoc_uptime_seconds gauge", f"madoc_uptime_seconds {time.time() - started:.0f}",
             "# TYPE madoc_queue_length gauge", f"madoc_queue_length {point['queue']}",
             "# TYPE madoc_models_loaded gauge", f"madoc_models_loaded {len(loaded_models())}"]
    for name, key in (("madoc_cpu_percent", "cpu"), ("madoc_memory_used_bytes", "memory_used"),
                      ("madoc_memory_total_bytes", "memory_total"), ("madoc_process_memory_bytes", "process_memory"),
                      ("madoc_disk_used_bytes", "disk_used"), ("madoc_disk_total_bytes", "disk_total")):
        if point.get(key) is not None:
            lines += [f"# TYPE {name} gauge", f"{name} {point[key]}"]
    for name, key, scale in (("madoc_gpu_utilization_percent", "util", 1), ("madoc_gpu_temperature_celsius",
                             "temperature", 1), ("madoc_gpu_memory_used_bytes", "memory_used", 1024 * 1024),
                             ("madoc_gpu_memory_total_bytes", "memory_total", 1024 * 1024),
                             ("madoc_gpu_power_watts", "power", 1)):
        rows = [f'{name}{{gpu="{gpu["index"]}"}} {gpu[key] * scale}' for gpu in point["gpus"] if gpu[key] is not None]
        if rows:
            lines += [f"# TYPE {name} gauge", *rows]
    lines += ["# TYPE madoc_step_seconds summary"]
    for (step, ok), (count, seconds) in sorted(totals.items()):
        labels = f'step="{step}",ok="{str(ok).lower()}"'
        lines += [f"madoc_step_seconds_count{{{labels}}} {count}", f"madoc_step_seconds_sum{{{labels}}} {seconds:.3f}"]
    return "\n".join(lines) + "\n"

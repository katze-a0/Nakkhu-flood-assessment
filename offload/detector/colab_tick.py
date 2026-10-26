# Colab tick: launcher alive?, finished units, log tails, GPU use (one JSON line first).
import json
import os
import subprocess
from pathlib import Path

work = Path("/content/nkwork")
pidf = work / "launcher.pid"
pid = int(pidf.read_text()) if pidf.exists() else None
alive = False
if pid is not None and os.path.exists(f"/proc/{pid}"):
    alive = "zombie" not in open(f"/proc/{pid}/status").read().lower()
done = sorted((work / "nk" / "done").glob("*.json")) if (work / "nk" / "done").exists() else []
print(json.dumps({"alive": alive, "done": len(done), "finished": (work / "finished.json").exists(),
                  "units": [json.loads(p.read_text())["status"] for p in done]}))
for f in sorted((work / "logs").glob("*.txt")) + [work / "setup.txt", work / "launcher_stdout.txt"]:
    if f.exists():
        tail = f.read_text(errors="replace").strip().splitlines()[-2:]
        print("==", f.name, "|", " / ".join(tail)[-300:].encode("ascii", "replace").decode())
print(subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
                     shell=True, capture_output=True, text=True).stdout.strip())

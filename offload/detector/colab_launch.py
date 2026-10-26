# Colab: start the plan 007 launcher detached (nohup) on /content/nkin; args from env NK_ARGS.
import os
import subprocess
from pathlib import Path

inp = Path("/content/nkin")
work = Path("/content/nkwork")
work.mkdir(exist_ok=True)
import zipfile  # noqa: E402

with zipfile.ZipFile(inp / "code.zip") as z:
    z.extractall("/content/nkcode")
args = os.environ["NK_ARGS"].split()
log = open(work / "launcher_stdout.txt", "a")
p = subprocess.Popen(["nohup", "python3", "/content/nkcode/code/launcher.py", "--input",
                      str(inp), "--work", str(work), *args], stdout=log,
                     stderr=subprocess.STDOUT, start_new_session=True)
(work / "launcher.pid").write_text(str(p.pid))
print("started", p.pid, args)

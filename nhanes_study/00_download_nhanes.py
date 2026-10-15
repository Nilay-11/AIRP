"""Download the NHANES 2011-2018 files used in this study from the CDC into data_raw/."""
import time
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data_raw"
RAW.mkdir(exist_ok=True)
CYCLES = {"2011": "G", "2013": "H", "2015": "I", "2017": "J"}
FILES = ["DEMO", "BPX", "BMX", "MCQ", "TCHOL", "HDL", "GHB", "BIOPRO", "SMQ", "DIQ", "BPQ"]

for year, s in CYCLES.items():
    for f in FILES:
        out = RAW / f"{f}_{s}.xpt"
        if out.exists():
            continue
        url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{f}_{s}.xpt"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        out.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        print("downloaded", out.name)
        time.sleep(0.5)
print("done")

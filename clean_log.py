"""
Cleans up data/drowsiness_log.csv by removing old fake/demo rows that got
mixed in from the notebook's old fallback code, keeping only REAL rows
logged by src/drowsiness_detector.py (or drowsiness_detector_simple.py).

Real rows always look like this (4 columns, ISO timestamp with a "T"):
    2026-08-02T21:06:30,DROWSINESS_ALERT,0.229,2.133

Fake/demo rows look like this (3 columns, plain date+time, no "T"):
    01-01-2026 09:04,0.222466828,DROWSY

Run this from the project ROOT folder:
    python clean_log.py
"""

import pandas as pd
import os

LOG_PATH = os.path.join("data", "drowsiness_log.csv")
BACKUP_PATH = os.path.join("data", "drowsiness_log_BACKUP.csv")

if not os.path.exists(LOG_PATH):
    print(f"No file found at {LOG_PATH} - nothing to clean.")
    raise SystemExit

# Always back up the original file first, just in case.
import shutil
shutil.copy(LOG_PATH, BACKUP_PATH)
print(f"Backed up original file to {BACKUP_PATH}")

# Read the file as plain text lines, since the two formats have a
# different number of columns and can't both be parsed by pd.read_csv
# in one pass.
with open(LOG_PATH, "r") as f:
    lines = f.readlines()

real_rows = []
header = "timestamp,event,ear,mar\n"

for line in lines:
    line = line.strip()
    if not line:
        continue
    if line.startswith("timestamp,event,ear,mar"):
        continue  # skip an existing header if present
    parts = line.split(",")
    # Real rows always have exactly 4 parts AND the timestamp contains "T"
    # (ISO format, e.g. 2026-08-02T21:06:30), which fake demo rows never have.
    if len(parts) == 4 and "T" in parts[0]:
        real_rows.append(line)

with open(LOG_PATH, "w") as f:
    f.write(header)
    for row in real_rows:
        f.write(row + "\n")

print(f"Cleaned! Kept {len(real_rows)} real rows.")
print(f"Removed {len(lines) - len(real_rows)} old fake/demo rows.")
print(f"Your real session data is now safely in {LOG_PATH}")

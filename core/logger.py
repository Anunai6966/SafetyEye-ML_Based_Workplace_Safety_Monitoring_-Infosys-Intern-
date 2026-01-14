# core/logger.py
import os, csv
from datetime import datetime
import cv2

LOG_DIR = "logs"
SNAP_DIR = os.path.join(LOG_DIR, "violations")
CSV_PATH = os.path.join(LOG_DIR, "events.csv")

os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 🔑 SINGLE, STANDARD SCHEMA
FIELDNAMES = [
    "timestamp",
    "person_id",
    "missing",
    "people_count",
    "violations_count",
]

# create CSV with correct headers if missing
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDNAMES)


def log_violation(person_idx, missing_helmet, missing_vest, conf_summary, bbox, frame):
    """
    Save snapshot and append standardized CSV row.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ----- snapshot -----
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    crop = frame[y1c:y2c, x1c:x2c] if (x2c > x1c and y2c > y1c) else frame

    snap_name = f"{ts.replace(':','')}_p{person_idx}.jpg"
    snap_path = os.path.join(SNAP_DIR, snap_name)
    cv2.imwrite(snap_path, crop)

    # ----- standardized log -----
    missing = []
    if missing_helmet:
        missing.append("helmet")
    if missing_vest:
        missing.append("vest")

    row = [
        ts,
        f"person_{person_idx}" if person_idx >= 0 else "",
        ",".join(missing),
        0,      # people_count (used by analytics)
        1       # violations_count
    ]

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return snap_path

# screenshot_stitcher\progress.py
import sys


def _fmt_hms(sec: float) -> str:
    msec = int((sec - int(sec)) * 1000)
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{msec:03d}"


def _print_progress(curr, total, prefix="", bar_len=30):
    ratio = 0 if total <= 0 else min(max(curr / total, 0), 1)
    filled = int(bar_len * ratio)
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = int(ratio * 100)
    sys.stdout.write(f"\r{prefix} [{bar}] {pct:3d}%")
    sys.stdout.flush()
    if curr >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()

# screenshot_stitcher\pipeline.py
import time
from .preprocess import _prep_gray_masks
from .overlap import find_overlap_gray
from .progress import _print_progress


def _accumulate_positions(
    imgs,
    *,
    direction,
    max_shift_ratio,
    tol,
    conf_min,
    slack_frac,
    early_stop,
    min_valid_frac,
):
    """인접 페어 오프셋 누적 → 각 이미지의 절대좌표 리스트 반환"""
    gm = _prep_gray_masks(imgs)
    positions = [(0, 0)]
    pair_confs = []
    n = len(imgs) - 1
    for i in range(n):
        title = f"[INFO] ({i+1}/{n}) overlap {i} -> {i+1}"
        print(title)

        (grayA, validA) = gm[i]
        (grayB, validB) = gm[i + 1]

        # 진행률 바
        def _cb(pct):
            _print_progress(pct, 100, prefix="   progress")

        t0 = time.time()

        dx, dy, conf = find_overlap_gray(
            grayA,
            validA,
            grayB,
            validB,
            max_shift_ratio=max_shift_ratio,
            tol=tol,
            direction=direction,
            slack_frac=slack_frac,
            progress_cb=_cb,
            early_stop=early_stop,
            min_valid_frac=min_valid_frac,
        )
        dt = time.time() - t0

        if conf < conf_min:
            print(
                f"[WARN] low confidence {conf:.3f} at pair {i}->{i+1}, time={dt:.2f}s"
            )
        else:
            print(
                f"[OK] dx={dx}, dy={dy}, conf={conf:.3f}, pair={i}->{i+1}, time={dt:.2f}s"
            )
        pair_confs.append(conf)
        px, py = positions[-1]
        positions.append((px + dx, py + dy))

    return positions, gm, pair_confs

import argparse
import os
import re
import sys
from pathlib import Path
import numpy as np
import cv2
import time


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


def _natural_key(s: str):
    # 'img_2.png' < 'img_10.png' 를 지키도록 정렬
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_images(input_dir):
    """입력 폴더 내의 PNG 파일 경로를 정렬하여 리스트로 반환"""
    p = Path(input_dir)
    if not p.is_dir():
        print(f"[에러] 입력 폴더가 없어요: {input_dir}")
        sys.exit(1)

    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".png"]
    if not files:
        print(f"[에러] {input_dir}에 PNG 이미지가 없어요")
        sys.exit(1)

    if len(files) < 2:
        print(f"[에러] 스티칭에는 최소 2장이 필요해요(현재 {len(files)}장).")
        sys.exit(1)

    files.sort(key=lambda x: _natural_key(x.name))

    paths = [str(f) for f in files]
    print(f"[INFO] 이미지 로드: {len(paths)}장")
    return paths


def _read_cv_images(paths):
    """경로 리스트 -> cv2 이미지 리스트 (BGRA 유지)"""
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[에러] cv2로 이미지 읽기 실패: {p}")
            sys.exit(1)
        imgs.append(img)
    return imgs


def _prep_gray_masks(imgs):
    """이미지들을 (gray, valid_mask)로 한 번만 변환해 캐시"""
    out = []
    for img in imgs:
        out.append(_to_gray_and_mask(img))
    return out


def _to_gray_and_mask(img):
    """
    img: BGR 또는 BGRA (np.uint8)
    반환: gray(H,W), valid_mask(H,W) (True=비교대상)
    """

    # ndim은 배열의 차원 수 (H,W) -> 2, (H,W,3) -> 3, (H,W,4) -> 3
    # shape는 배열의 크기를 담은 튜플
    # gray: shape == (H,W)
    # color: shape == (H,W,3)
    # BGRA: shape == (H,W,4)
    if img.ndim == 2:  # gray
        gray = img
        valid = np.ones_like(gray, dtype=bool)
        return gray, valid

    if img.shape[2] == 4:  # BGRA
        bgr = img[:, :, :3]  # :3은 BGR 채널만 취함 -> 알파 채널 제외
        alpha = img[:, :, 3]  # 3은 알파 채널만 취함 -> BGR 채널 제외
        gray = cv2.cvtColor(
            bgr, cv2.COLOR_BGR2GRAY
        )  # 0.299*R + 0.587*G + 0.114*B 가중합 (사람 시야 보정)
        valid = alpha > 0
        return gray, valid

    else:  # BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        valid = np.ones(gray.shape, dtype=bool)
        return gray, valid


def _overlap_slices(H1, W1, H2, W2, dx, dy):
    """
    B를 A 위에 (dx, dy)만큼 이동했을 때 A/B의 겹침 슬라이스 계산.
    좌표계: A 기준. (0,0) = 좌상단
    """
    # A에서의 겹침 영역 (H1, W1)
    ax1 = max(0, dx)  # 좌상단 꼭짓점
    ay1 = max(0, dy)  # 좌상단 꼭짓점
    ax2 = min(W1, dx + W2)  # 우하단 꼭짓점
    ay2 = min(H1, dy + H2)  # 우하단 꼭짓점

    if ax1 >= ax2 or ay1 >= ay2:
        return None

    # B에서의 대응 영역 (ax, ay) 와 dx dy 차이가 남
    ow = ax2 - ax1
    oh = ay2 - ay1
    bx1 = ax1 - dx
    by1 = ay1 - dy
    bx2 = bx1 + ow
    by2 = by1 + oh
    return (slice(ay1, ay2), slice(ax1, ax2), slice(by1, by2), slice(bx1, bx2))


def _overlap_area(H1, W1, H2, W2, dx, dy):
    """서로 겹치는 면적 계산"""
    ax1 = max(0, dx)
    ay1 = max(0, dy)
    ax2 = min(W1, dx + W2)
    ay2 = min(H1, dy + H2)
    ow = ax2 - ax1
    oh = ay2 - ay1
    if ow <= 0 or oh <= 0:
        return 0
    return ow * oh


def _ordered_candidates(H1, W1, H2, W2, min_dx, max_dx, min_dy, max_dy):
    """겹치는 면적 내림차순 정렬"""
    cand = []
    for dy in range(min_dy, max_dy + 1):
        for dx in range(min_dx, max_dx + 1):
            area = _overlap_area(H1, W1, H2, W2, dx, dy)
            if area > 0:
                cand.append((dx, dy, area))
    # 면적 내림차순, 면적 같으면 중심(L1)이 가까운 순
    cand.sort(key=lambda t: (-t[2], abs(t[0]) + abs(t[1])))
    return cand


def _score_at(grayA, validA, grayB, validB, dx, dy, tol):
    """
    (dx,dy)에서의 일치도(confidence) 계산.
    confidence = (abs(A-B)<=tol & 유효마스크)/유효겹침픽셀
    """
    H1, W1 = grayA.shape
    H2, W2 = grayB.shape
    sl = _overlap_slices(H1, W1, H2, W2, dx, dy)  # 슬라이스 인덱스 4개 반환
    if sl is None:
        return 0.0, 0

    ay, ax, by, bx = sl
    a = grayA[ay, ax]
    b = grayB[by, bx]
    v = validA[ay, ax] & validB[by, bx]  # 투명한지 체크
    if not v.any():
        return 0.0, 0

    diff = cv2.absdiff(a, b)  # 절댓값 계산
    match = (diff <= tol) & v  # 차이가 기준 이내이며 유효 픽셀인지
    num_valid = int(v.sum())  # 비교한 유효 픽셀 갯수
    num_match = int(match.sum())  # 유효 픽셀 중 기준에 적합한 갯수
    conf = num_match / num_valid  # 비율 계산
    return conf, num_valid


def find_overlap_gray(
    grayA,
    validA,
    grayB,
    validB,
    max_shift_ratio=1,
    tol=5,
    direction="both",
    slack_frac=0.25,  # 흔들림 허용 비율 (기본 25%)
    progress_cb=None,
    early_stop=True,
    min_valid_frac=0.75,
):
    """
    완전 브루트포스로 (dx,dy,confidence) 찾기.
    - imgA, imgB 크기 달라도 OK(배율만 동일 가정)
    direction이 'vertical'이면 dx에 ±slack, 'horizontal'이면 dy에 ±slack 허용.
    """

    H1, W1 = grayA.shape  # gray는 2차원
    H2, W2 = grayB.shape

    # 탐색 한계: 각각의 크기에 비례 -> 겹치는 영역의 최대
    min_dx = int(-W2 * max_shift_ratio)
    max_dx = int(W1 * max_shift_ratio)
    min_dy = int(-H2 * max_shift_ratio)
    max_dy = int(H1 * max_shift_ratio)

    # 축 제한
    if direction == "vertical":  # 가로 이동 제한
        # 가로 정렬 범위(포함 정렬) ± 슬랙
        nominal_min = min(0, W1 - W2)
        nominal_max = max(0, W1 - W2)
        slack = int(round(W2 * slack_frac))
        min_dx = nominal_min - slack
        max_dx = nominal_max + slack
    elif direction == "horizontal":  # 세로 이동 제한
        # 세로 정렬 범위(포함 정렬) ± 슬랙
        nominal_min = min(0, H1 - H2)
        nominal_max = max(0, H1 - H2)
        slack = int(round(H2 * slack_frac))
        min_dy = nominal_min - slack
        max_dy = nominal_max + slack

    # 후보를 겹침 면적 내림차순으로 준비
    candidates = _ordered_candidates(H1, W1, H2, W2, min_dx, max_dx, min_dy, max_dy)
    if not candidates:
        return 0, 0, 0.0

    total = len(candidates)
    last_report = -1

    # 탐색
    best_conf = -1.0
    best_dx = best_dy = 0
    best_valid = 0

    for i, (dx, dy, area) in enumerate(candidates):

        if progress_cb:
            # 너무 잦은 호출 방지: 1% 단위로만 보고
            pct = int((i + 1) * 100 / total) if total > 0 else 100
            if pct != last_report:
                progress_cb(pct)
                last_report = pct

        conf, num_valid = _score_at(grayA, validA, grayB, validB, dx, dy, tol)
        if (
            conf > best_conf
            or (conf == best_conf and num_valid > best_valid)
            or (
                conf == best_conf
                and num_valid == best_valid
                and (abs(dx) + abs(dy) < abs(best_dx) + abs(best_dy))
            )
        ):  # 일치율 > 유효 픽셀 수 > 더 적은 이동
            best_conf, best_valid, best_dx, best_dy = conf, num_valid, dx, dy

        # 조기 종료: conf==1 이면서 '충분히 큰' 겹침
        if early_stop and conf == 1.0 and num_valid >= int(min_valid_frac * area):
            return int(dx), int(dy), float(conf)

    return int(best_dx), int(best_dy), float(best_conf)


def _ensure_bgra(img):
    """입력(img: GRAY/BGR/BGRA)을 BGRA로 통일(알파=255)"""
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        a = np.full(img.shape, 255, np.uint8)
        return np.dstack([bgr, a])
    if img.shape[2] == 3:  # BGR
        a = np.full(img.shape[:2], 255, np.uint8)
        return np.dstack([img, a])
    return img  # 이미 BGRA


def _paste_bgra(canvas, src, x, y):
    """src(bgra)를 canvas의 (x, y)에 클리핑해 덮어쓰기"""
    Hc, Wc = canvas.shape[:2]  # canvas
    Hs, Ws = src.shape[:2]  # src

    # canvas 경계 내로 클리핑
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(Wc, x + Ws), min(Hc, y + Hs)
    if x1 >= x2 or y1 >= y2:
        return  # 만에 하나 겹침 없음

    sx1, sy1 = x1 - x, y1 - y  # src를 기준으로 했을 때 좌상단 좌표
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)  # src를 기준으로 했을 때 우하단 좌표

    canvas[y1:y2, x1:x2, :3] = src[
        sy1:sy2, sx1:sx2, :3
    ]  # 덮어쓰기(추후에 베젤 관련 작업 예정)
    canvas[y1:y2, x1:x2, 3] = 255  # 알파채널은 255로 통일


def _accumulate_positions(
    imgs,
    *,
    direction,
    max_shift_ratio=1,
    tol,
    conf_min=0.05,
    slack_frac=0.25,
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


def _stitch_all(imgs, positions):
    """절대좌표에 맞춰 모두 붙여 하나의 BGRA 반환"""
    xs, ys = [], []
    for img, (x, y) in zip(imgs, positions):
        h, w = img.shape[:2]
        xs += [x, x + w]
        ys += [y, y + h]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    W = max_x - min_x
    H = max_y - min_y
    shift_x, shift_y = -min_x, -min_y

    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    for img, (x, y) in zip(imgs, positions):
        _paste_bgra(canvas, _ensure_bgra(img), x + shift_x, y + shift_y)
    return canvas


def _stitch_all_distance(imgs, positions):
    # 전역 바운딩
    xs, ys = [], []
    for img, (x, y) in zip(imgs, positions):
        h, w = img.shape[:2]
        xs += [x, x + w]
        ys += [y, y + h]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    W = max_x - min_x
    H = max_y - min_y
    shift_x, shift_y = -min_x, -min_y

    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    owner = np.full((H, W), -1, dtype=np.int32)

    # 전역 중심 좌표
    centers = []
    for img, (x, y) in zip(imgs, positions):
        h, w = img.shape[:2]
        cx = (x + w / 2) + shift_x
        cy = (y + h / 2) + shift_y
        centers.append((cx, cy))

    for idx, (img, (x, y)) in enumerate(zip(imgs, positions)):
        src = _ensure_bgra(img)
        h, w = src.shape[:2]
        gx, gy = x + shift_x, y + shift_y  # 전역 좌표

        # 캔버스 클리핑
        x1, y1 = max(0, gx), max(0, gy)
        x2, y2 = min(W, gx + w), min(H, gy + h)
        if x1 >= x2 or y1 >= y2:
            continue

        # 새로 칠할 영역
        subW, subH = (x2 - x1), (y2 - y1)
        c_sub = canvas[y1:y2, x1:x2, :4]
        o_sub = owner[y1:y2, x1:x2]
        s_sub = src[y1 - gy : y1 - gy + subH, x1 - gx : x1 - gx + subW, :4]

        # 유효(알파>0) 마스크
        valid_sub = s_sub[:, :, 3] > 0
        empty_mask_pre = (o_sub == -1) & valid_sub  # 완전 빈 곳
        overlap_mask_pre = (o_sub != -1) & valid_sub  # 기존 오너가 있던 곳과 겹침

        # 겹치는 기존 오너 id 집합
        uniq = np.unique(o_sub[overlap_mask_pre])
        # 방어적 제외(실제로는 필요 없지만 안전)
        uniq = uniq[uniq != idx]

        # 빈 곳은 바로 채움
        if empty_mask_pre.any():
            c_sub[empty_mask_pre] = s_sub[empty_mask_pre]
            o_sub[empty_mask_pre] = idx

        # 겹침은 거리 기반 선택
        if uniq.size:

            # 전역 좌표 그리드 (오너 중심 거리 계산용)
            yy, xx = np.mgrid[y1:y2, x1:x2]
            xx = xx.astype(np.float32)
            yy = yy.astype(np.float32)

            cx_i, cy_i = centers[idx]  # 미리 계산해 둔 각 이미지 중심 (전역 좌표)
            d_cur = (xx - cx_i) ** 2 + (yy - cy_i) ** 2  # 현재(src)까지의 거리^2

            for k in uniq:
                # 아직 k가 소유하고 있고 이번 src도 덮는 픽셀만 대상으로
                mk = (o_sub == k) & overlap_mask_pre
                if not mk.any():
                    continue
                cx_k, cy_k = centers[k]
                d_k = (xx - cx_k) ** 2 + (yy - cy_k) ** 2
                repl = mk & (d_cur < d_k)
                if repl.any():
                    c_sub[repl] = s_sub[repl]
                    o_sub[repl] = idx

    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description="Screenshot Stitcher MVP")
    parser.add_argument("--input", type=str, help="입력 이미지 폴더 경로")
    parser.add_argument(
        "--output",
        type=str,
        default="stitched.png",
        help="출력 파일명 (기본: stitched.png)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="vertical",
        choices=["both", "vertical", "horizontal"],
        help="겹침 방향 고정 (both/vertical/horizontal)",
    )
    parser.add_argument(
        "--max-shift-ratio",
        type=float,
        default=1.0,
        help="탐색 범위 비율(각 축별 A/B 크기에 비례). vertical/horizontal에서는 비제한 축에만 주로 영향.",
    )
    parser.add_argument(
        "--tol", type=int, default=5, help="픽셀 차이 허용치(그레이스케일)"
    )
    parser.add_argument(
        "--conf-min", type=float, default=0.05, help="페어 매칭 최소 신뢰도 경고 임계치"
    )
    parser.add_argument(
        "--slack-frac", type=float, default=0.25, help="수직/수평 외축 흔들림 허용 비율"
    )
    parser.add_argument(
        "--no-early-stop",
        dest="early_stop",
        action="store_false",
        help="완전일치(conf=1) 조기 종료 끔",
    )
    parser.add_argument(
        "--early-stop",
        dest="early_stop",
        action="store_true",
        help="완전일치(conf=1) 조기 종료 켬(기본)",
    )
    parser.set_defaults(early_stop=True)

    parser.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.75,
        help="조기 종료 시 요구되는 겹침 면적 비율(0~1)",
    )

    return parser.parse_args()


def main():
    try:
        args = parse_args()

        # --- 전체 타이머 시작 ---
        t0_all = time.perf_counter()

        # 1. 입력 받기
        if args.input:
            input_dir = os.path.abspath(args.input)
        else:
            # stitch.py가 있는 위치 + images 폴더
            script_dir = os.path.dirname(os.path.abspath(__file__))
            input_dir = os.path.join(script_dir, "images")

        # 2. 이미지 불러오기

        image_files = load_images(
            input_dir
        )  # 해당 경로에 있는 모든 이미지 파일 경로 리스트
        imgs = _read_cv_images(image_files)  # 이미지 데이터(numpy arr) 리스트

        # 3. 이미지 겹침 계산
        t0_match = time.perf_counter()
        positions, gm, pair_confs = _accumulate_positions(
            imgs,
            direction=args.direction,
            max_shift_ratio=args.max_shift_ratio,
            tol=args.tol,
            conf_min=args.conf_min,
            slack_frac=args.slack_frac,
            early_stop=args.early_stop,
            min_valid_frac=args.min_valid_frac,
        )
        t1_match = time.perf_counter()

        if all(c == 1.0 for c in pair_confs):  # 4-1. 덮어쓰며 이어 붙이기
            print("[INFO] all pairs perfectly matched (conf=1.0). using fast paste.")
            stitched = _stitch_all(imgs, positions)
        else:  # 4-2. 베젤 제거하며 이어 붙이기
            print(
                "[INFO] imperfect matches detected. using distance-based anti-bezel stitch."
            )
            stitched = _stitch_all_distance(imgs, positions)

        # 5. 출력 저장
        out_path = os.path.abspath(args.output)
        ok = cv2.imwrite(out_path, stitched)
        if not ok:
            print(f"[ERROR] 저장 실패: {out_path}")
        print(f"[OK] saved -> {out_path}  size={stitched.shape[1]}x{stitched.shape[0]}")

        # --- 시간 출력 ---
        total_sec = time.perf_counter() - t0_all
        match_sec = t1_match - t0_match
        print(f"[TIME] match={_fmt_hms(match_sec)}, total={_fmt_hms(total_sec)}")

    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단(Ctrl+C). 중간 진행 상태에서 종료합니다.")

    pass


if __name__ == "__main__":
    main()

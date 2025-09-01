import argparse
import os
import re
import sys
from pathlib import Path
import numpy as np
import cv2


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


def find_overlap(
    imgA, imgB, max_shift_ratio=1, tol=5, coarse_stride=1, refine_window=18
):
    """
    완전 브루트포스(거친탐색→정밀)로 (dx,dy,confidence) 찾기.
    - imgA, imgB 크기 달라도 OK(배율만 동일 가정)
    """
    grayA, validA = _to_gray_and_mask(imgA)
    grayB, validB = _to_gray_and_mask(imgB)

    H1, W1 = grayA.shape  # gray는 2차원
    H2, W2 = grayB.shape

    # 탐색 한계: 각각의 크기에 비례 -> 겹치는 영역의 최대
    max_dx = int(min(W1, W2) * max_shift_ratio)
    max_dy = int(min(H1, H2) * max_shift_ratio)

    best_conf = -1.0
    best_dx = 0
    best_dy = 0
    best_valid = 0

    # 1) 거친 탐색
    for dy in range(-max_dy, max_dy + 1, coarse_stride):
        for dx in range(-max_dx, max_dx + 1, coarse_stride):
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

    # 2) 정밀 탐색
    dx1 = max(-max_dx, best_dx - refine_window)
    dx2 = min(+max_dx, best_dx + refine_window)
    dy1 = max(-max_dy, best_dy - refine_window)
    dy2 = min(+max_dy, best_dy + refine_window)
    for dy in range(dy1, dy2 + 1):
        for dx in range(dx1, dx2 + 1):
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


def _accumulate_positions(imgs):
    """인접 페어 오프셋 누적 → 각 이미지의 절대좌표 리스트 반환"""
    positions = [(0, 0)]
    for i in range(len(imgs) - 1):
        print(f"[INFO] find_overlap start: {i} -> {i+1}")
        dx, dy, conf = find_overlap(imgs[i], imgs[i + 1])
        print(f"[INFO] find_overlap done : {i} -> {i+1}")
        # print(f"[OVERLAP] {i}->{i+1} dx={dx}, dy={dy}, conf={conf:.3f}")
        px, py = positions[-1]
        positions.append((px + dx, py + dy))
    return positions


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


def parse_args():
    parser = argparse.ArgumentParser(description="Screenshot Stitcher MVP")
    parser.add_argument("--input", type=str, help="입력 이미지 폴더 경로")
    parser.add_argument(
        "--output",
        type=str,
        default="stitched.png",
        help="출력 파일명 (기본: stitched.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 입력 받기
    if args.input:
        input_dir = os.path.abspath(args.input)
    else:
        # stitch.py가 있는 위치 + images 폴더
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, "images")

    # 2. 이미지 불러오기
    # 해당 경로에 있는 모든 이미지 파일 경로 리스트
    image_files = load_images(input_dir)
    # 이미지 데이터(numpy arr) 리스트
    imgs = _read_cv_images(image_files)

    # 3. 이미지 겹침 계산
    positions = _accumulate_positions(imgs)

    # 4. 이어 붙이기
    stitched = _stitch_all(imgs, positions)

    # 5. 이어붙인 이미지 베젤 제거
    # 6. 출력 저장
    out_path = os.path.abspath(args.output)
    cv2.imwrite(out_path, stitched)
    print(f"[OK] saved -> {out_path}  size={stitched.shape[1]}x{stitched.shape[0]}")

    pass


if __name__ == "__main__":
    main()

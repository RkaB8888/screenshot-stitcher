# screenshot_stitcher\overlap.py
import numpy as np
import cv2


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
        if conf > 0.9 and (
            (num_valid > best_valid)
            or (
                num_valid == best_valid
                and (abs(dx) + abs(dy) < abs(best_dx) + abs(best_dy))
            )
        ):  # 일치율 > 유효 픽셀 수 > 더 적은 이동
            best_conf, best_valid, best_dx, best_dy = conf, num_valid, dx, dy

        # 조기 종료: conf==1 이면서 '충분히 큰' 겹침
        if early_stop and conf == 1.0 and num_valid >= int(min_valid_frac * area):
            return int(dx), int(dy), float(conf)

    return int(best_dx), int(best_dy), float(best_conf)

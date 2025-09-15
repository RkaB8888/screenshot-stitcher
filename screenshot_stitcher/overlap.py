# screenshot_stitcher\overlap.py
import numpy as np
import cv2


def _inner_mask(H, W, left, top, right, bottom):
    """
    베젤을 제외한 내부만 True인 마스크 반환.
    베젤 합이 크거나 영역이 비정상이면 빈 마스크 처리.
    """
    m = np.zeros((H, W), dtype=bool)
    x1 = max(0, left)
    y1 = max(0, top)
    x2 = max(x1, W - max(0, right))
    y2 = max(y1, H - max(0, bottom))
    if x1 < x2 and y1 < y2:
        m[y1:y2, x1:x2] = True
    return m


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


def _score_at(
    grayA,
    validA,
    grayB,
    validB,
    dx,
    dy,
    tol,
    sample_step=4,
    eps=1e-6,
    bezA=None,
    bezB=None,
):
    """
    (dx,dy)에서의 단순 일치도 계산.
    - 픽셀 절대차가 tol 이하면 1점, 아니면 0점
    - score = 일치하는 유효 샘플 수
    - norm  = 유효 샘플 수 (이론 최대)
    - conf  = score / norm  (0~1)
    """
    H1, W1 = grayA.shape
    H2, W2 = grayB.shape
    sl = _overlap_slices(H1, W1, H2, W2, dx, dy)  # 슬라이스 인덱스 4개 반환
    if sl is None:
        return 0.0, 0, 0.0, 0.0

    ay, ax, by, bx = sl

    # 그리드 서브샘플링 슬라이스 구성
    sAy = slice(ay.start, ay.stop, sample_step)
    sAx = slice(ax.start, ax.stop, sample_step)
    sBy = slice(by.start, by.stop, sample_step)
    sBx = slice(bx.start, bx.stop, sample_step)

    # 원래 전체 겹침 → 샘플 겹침으로 다운샘플
    a = grayA[sAy, sAx]
    b = grayB[sBy, sBx]
    v = validA[sAy, sAx] & validB[sBy, sBx]  # 투명한지 체크

    # 베젤 제외: 내부마스크가 주어졌다면 샘플 슬라이스로 잘라서 v에 AND
    if bezA is not None:
        v &= bezA[sAy, sAx]
    if bezB is not None:
        v &= bezB[sBy, sBx]

    if not v.any():
        return 0.0, 0, 0.0, 0.0

    # 픽셀 일치도
    diff = cv2.absdiff(a, b)
    match = (diff <= tol) & v

    # 집계: 총점(score)과 정규화(conf)
    score = float(match.sum())  # 면적 반영된 실제 점수(타이브레이커 강함)
    norm = float(v.sum()) + eps  # 전체 유효 샘플 수
    conf = score / norm  # 0~1 스케일의 신뢰도
    num_valid = int(v.sum())  # 비교한 유효 픽셀 갯수
    return conf, num_valid, score, norm


def find_overlap_gray(
    grayA,
    validA,
    grayB,
    validB,
    max_shift_ratio=1,
    tol=0,
    direction="both",
    slack_frac=0.25,  # 흔들림 허용 비율 (기본 25%)
    progress_cb=None,
    sample_step=4,
    bezel=(0, 0, 0, 0),
):
    """
    완전 브루트포스로 (dx,dy,confidence) 찾기.
    - imgA, imgB 크기 달라도 OK(배율만 동일 가정)
    direction이 'vertical'이면 dx에 ±slack, 'horizontal'이면 dy에 ±slack 허용.
    """

    H1, W1 = grayA.shape  # gray는 2차원
    H2, W2 = grayB.shape

    # 베젤 → 내부마스크 (A, B는 같은 베젤이라고 가정한 경우)
    bz_left, bz_top, bz_right, bz_bottom = bezel
    bezA = _inner_mask(H1, W1, bz_left, bz_top, bz_right, bz_bottom)
    bezB = _inner_mask(H2, W2, bz_left, bz_top, bz_right, bz_bottom)

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
        return 0, 0, 0.0, 0.0, 1.0, True

    total = len(candidates)
    last_report = -1

    # 탐색 (가중 중심 튜플 정렬 + 폴백)
    best = None  # 임계/규칙 통과 기준으로 고른 최선
    best_overall = None  # 임계 무시, 단순 스코어 최대 후보(폴백용)

    for i, (dx, dy, area) in enumerate(candidates):

        if progress_cb:
            # 너무 잦은 호출 방지: 1% 단위로만 보고
            pct = int((i + 1) * 100 / total) if total > 0 else 100
            if pct != last_report:
                progress_cb(pct)
                last_report = pct

        conf, num_valid, score, norm = _score_at(
            grayA,
            validA,
            grayB,
            validB,
            dx,
            dy,
            tol,
            sample_step=sample_step,
            bezA=bezA,
            bezB=bezB,
        )

        key = (conf, score, num_valid, -(abs(dx) + abs(dy)))

        if best_overall is None or key > best_overall[0]:
            best_overall = (key, dx, dy, conf, score, norm)

        # 최선 갱신
        if best is None or key > best[0]:
            best = (key, dx, dy, conf, score, norm)

    # 최종 반환 (best 없으면 overall 폴백)
    if best is not None:
        _, dx, dy, conf, score, norm = best
        print("[INFO] return best")
        return int(dx), int(dy), float(conf), float(score), float(norm), False
    else:
        # 폴백 경고는 상위에서 로그로 출력하는 편이 깔끔
        _, dx, dy, conf, score, norm = best_overall
        print("[INFO] return best overall")
        return int(dx), int(dy), float(conf), float(score), float(norm), True

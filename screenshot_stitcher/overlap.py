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


def _center_roi_slices(ay, ax, by, bx, k=5):
    """겹침 슬라이스(ay,ax,by,bx)에서 중앙 k×k(가능하면) 슬라이스 반환"""
    oh = ay.stop - ay.start
    ow = ax.stop - ax.start
    if oh <= 0 or ow <= 0:
        return None
    r = max(1, k // 2)
    # 겹침이 작으면 k 축소
    hh = min(oh, 2 * r + 1)
    ww = min(ow, 2 * r + 1)
    cyA = ay.start + oh // 2
    cxA = ax.start + ow // 2
    cyB = by.start + oh // 2
    cxB = bx.start + ow // 2
    Ay = slice(max(ay.start, cyA - hh // 2), min(ay.stop, cyA + (hh - hh // 2)))
    Ax = slice(max(ax.start, cxA - ww // 2), min(ax.stop, cxA + (ww - ww // 2)))
    By = slice(max(by.start, cyB - hh // 2), min(by.stop, cyB + (hh - hh // 2)))
    Bx = slice(max(bx.start, cxB - ww // 2), min(bx.stop, cxB + (ww - ww // 2)))
    if (Ay.stop - Ay.start) <= 0 or (Ax.stop - Ax.start) <= 0:
        return None
    return Ay, Ax, By, Bx


def _ordered_candidates(H1, W1, H2, W2, min_dx, max_dx, min_dy, max_dy):
    """겹치는 면적 내림차순 정렬"""
    cand = []
    for dy in range(min_dy, max_dy + 1):
        for dx in range(min_dx, max_dx + 1):
            area = _overlap_area(H1, W1, H2, W2, dx, dy)
            if area > 0:
                cand.append((dx, dy, area))
    # 면적 내림차순, 면적 같으면 중심(L1)이 가까운 순
    # cand.sort(key=lambda t: (-t[2], abs(t[0]) + abs(t[1])))
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


# =========================
# 라인 기반 프리체크 + (옵션) 미니패치
# =========================
LINE_ATTEMPTS = 7  # 중앙선 포함 최대 시도 라인 수
LINE_STEP = 5  # 중앙에서 떨어진 간격(px)
MINI_K = 5  # 정밀 미니패치 크기(정수, 홀수 권장)


def _line_precheck(
    grayA,
    validA,
    grayB,
    validB,
    ay,
    ax,
    by,
    bx,
    tol,
    bezA=None,
    bezB=None,
    *,
    use_rows=False,  # True: 행 비교, False: 열 비교
):
    """
    행/열 단일선 비교 프리체크.
    - 균일선(B가 단일값) 스킵
    - 유효 픽셀 없는 선 스킵
    - 한 픽셀이라도 |A-B| > tol 이면 False, 모두 ≤ tol 이면 True
    """
    # ROI 크기
    oh = ay.stop - ay.start
    ow = ax.stop - ax.start

    # 중앙 인덱스 및 탐색 오프셋 순서
    if use_rows:
        center = ay.start + oh // 2
    else:
        center = ax.start + ow // 2

    # 선 인덱스 시퀀스: 0, ±1, ±2 ... 에서 STEP 스케일
    offsets = [0]
    for t in range(1, LINE_ATTEMPTS):
        offsets.extend([t * LINE_STEP, -t * LINE_STEP])

    for off in offsets:
        if use_rows:
            yA = center + off
            yB = by.start + (yA - ay.start)
            if not (ay.start <= yA < ay.stop and by.start <= yB < by.stop):
                continue

            a_line = grayA[yA, ax]
            b_line = grayB[yB, bx]
            v_line = validA[yA, ax] & validB[yB, bx]
            if bezA is not None:
                v_line &= bezA[yA, ax]
            if bezB is not None:
                v_line &= bezB[yB, bx]
        else:
            xA = center + off
            xB = bx.start + (xA - ax.start)
            if not (ax.start <= xA < ax.stop and bx.start <= xB < bx.stop):
                continue

            a_line = grayA[ay, xA]
            b_line = grayB[by, xB]
            v_line = validA[ay, xA] & validB[by, xB]
            if bezA is not None:
                v_line &= bezA[ay, xA]
            if bezB is not None:
                v_line &= bezB[by, xB]

        if not np.any(v_line):
            continue

        # 균일선이면 정보 없음 → 다음 오프셋
        if np.ptp(b_line[v_line]) == 0:
            continue

        diff = np.abs(a_line.astype(np.int16) - b_line.astype(np.int16))

        # 이 선이 전부 tol 이내면 프리체크 통과
        if np.all(diff[v_line] <= tol):
            return True  # 통과
        else:
            return False  # 바로 탈락
        # 이 선에서 실패해도 다른 오프셋 선을 더 본다
        # (즉시 False 반환하지 않음)

    # 정보선 못 찾음 → 탈락으로 간주
    return False


def _mini_patch_check(
    grayA,
    validA,
    grayB,
    validB,
    ay,
    ax,
    by,
    bx,
    tol,
    bezA=None,
    bezB=None,
    k=MINI_K,
):
    """중앙 k×k 미니패치 모두 ≤ tol 이어야 통과(True). 유효 0이면 실패(False)."""
    sub = _center_roi_slices(ay, ax, by, bx, k=k)
    if sub is None:
        return False
    Ay, Ax, By, Bx = sub
    a = grayA[Ay, Ax]
    b = grayB[By, Bx]
    v = validA[Ay, Ax] & validB[By, Bx]
    if bezA is not None:
        v &= bezA[Ay, Ax]
    if bezB is not None:
        v &= bezB[By, Bx]
    if not np.any(v):
        return False
    diff = cv2.absdiff(a, b)
    return bool(np.all(diff[v] <= tol))


def _prelim_pass(
    grayA,
    validA,
    grayB,
    validB,
    dx,
    dy,
    tol,
    direction,
    bezA=None,
    bezB=None,
    *,
    prelim_refine=False,
):
    """
    후보 프리체크: 행/열 단일선 비교 + (옵션) 미니패치 AND
    direction에 따라 우선축 선택. both인 경우 행→열 순으로 1회 재시도.
    """
    H1, W1 = grayA.shape
    H2, W2 = grayB.shape
    sl = _overlap_slices(H1, W1, H2, W2, dx, dy)
    if sl is None:
        return False
    ay, ax, by, bx = sl

    def try_axis(use_rows: bool) -> bool:
        ok_line = _line_precheck(
            grayA,
            validA,
            grayB,
            validB,
            ay,
            ax,
            by,
            bx,
            tol,
            bezA=bezA,
            bezB=bezB,
            use_rows=use_rows,
        )
        if not ok_line:
            return False
        if prelim_refine:
            return _mini_patch_check(
                grayA,
                validA,
                grayB,
                validB,
                ay,
                ax,
                by,
                bx,
                tol,
                bezA=bezA,
                bezB=bezB,
                k=MINI_K,
            )
        return True

    if direction == "horizontal":
        if try_axis(use_rows=True):
            return True
        # 열로 1회 보조 시도
        return try_axis(use_rows=False)

    if direction == "vertical":
        if try_axis(use_rows=False):
            return True
        # 행으로 1회 보조 시도
        return try_axis(use_rows=True)

    # both: 열 우선 → 행 보조
    if try_axis(use_rows=False):
        return True
    return try_axis(use_rows=True)


def find_overlap_gray(
    grayA,
    validA,
    grayB,
    validB,
    max_shift_ratio=1,
    tol=0,
    direction="both",
    slack_frac=0.25,  # 흔들림 허용 비율 (기본 25%)
    progress_cb=None,  # 매칭(후보 선별) 진행 콜백
    progress_cb_refine=None,  # 정밀 계산 진행 콜백
    sample_step=4,
    bezel=(0, 0, 0, 0),
    prelim_refine=False,
):
    """
    완전 브루트포스로 (dx,dy,confidence) 찾기.
    - imgA, imgB 크기 달라도 OK(배율만 동일 가정)
    direction이 'vertical'이면 dx에 ±slack, 'horizontal'이면 dy에 ±slack 허용.
    """

    H1, W1 = grayA.shape
    H2, W2 = grayB.shape

    # 베젤 → 내부마스크
    bz_left, bz_top, bz_right, bz_bottom = bezel
    bezA = _inner_mask(H1, W1, bz_left, bz_top, bz_right, bz_bottom)
    bezB = _inner_mask(H2, W2, bz_left, bz_top, bz_right, bz_bottom)

    # 탐색 한계
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
        print("[WARN] NO CANDIDATE")
        # direction 기준 안전 오프셋 반환
        if direction == "horizontal":
            return int(W1), 0, 0.0, 0.0, 1.0, True
        else:
            return 0, int(H1), 0.0, 0.0, 1.0, True

    total = len(candidates)
    last_report = -1

    # 1) 프리체크(라인 기반 + 옵션 미니패치)
    prelim = []
    print(f"total: {total}")
    for i, (dx, dy, area) in enumerate(candidates):

        if progress_cb:
            # 너무 잦은 호출 방지: 1% 단위로만 보고
            pct = int((i + 1) * 100 / total) if total > 0 else 100
            if pct != last_report:
                progress_cb(pct)
                last_report = pct

        # 후보군 압축
        ok = _prelim_pass(
            grayA,
            validA,
            grayB,
            validB,
            dx,
            dy,
            tol,
            direction,
            bezA=bezA,
            bezB=bezB,
            prelim_refine=prelim_refine,
        )
        if ok:
            prelim.append((dx, dy, area))

    # 2) 프리체크 통과 후보 없음 → 겹침 없음 판단
    if not prelim:
        print("[WARN] NO PRELIM MATCH")
        # direction 기준 안전 오프셋
        if direction == "horizontal":
            return int(W1), 0, 0.0, 0.0, 1.0, True
        else:  # 'vertical' 또는 'both'는 세로로 이어붙임
            return 0, int(H1), 0.0, 0.0, 1.0, True

    # 3) 정밀 점수 계산
    final_best = None
    total_refine = len(prelim)
    last_report2 = -1
    print(f"total_refine: {total_refine}")
    for j, (dx, dy, area) in enumerate(prelim):
        if progress_cb_refine and total_refine > 0:
            pct2 = int((j + 1) * 100 / total_refine)
            if pct2 != last_report2:
                progress_cb_refine(pct2)
                last_report2 = pct2

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
        key = (conf, score, num_valid, area, -(abs(dx) + abs(dy)))
        if (final_best is None) or (key > final_best[0]):
            final_best = (key, dx, dy, conf, score, norm)

    _, dx, dy, conf, score, norm = final_best
    return int(dx), int(dy), float(conf), float(score), float(norm), False

# screenshot_stitcher\stitch.py
import numpy as np
import cv2


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

        if overlap_mask_pre.any():
            # 1) 값이 동일한 픽셀은 무조건 최신 owner 로 갱신 (색은 그대로)
            #    BGRA 전체 채널 동일 기준; RGB만 보려면 :3 로 바꿔도 됨.
            same_pix = overlap_mask_pre & (s_sub[:, :, :3] == c_sub[:, :, :3]).all(
                axis=2
            )
            if same_pix.any():
                o_sub[same_pix] = idx

            # 2) 나머지(값이 다른) 픽셀만 거리 비교
            remain = overlap_mask_pre & ~same_pix
            if remain.any():
                # 이 픽셀들에 대해 현재 존재하는 owner 후보들
                uniq = np.unique(o_sub[remain])
                # 전역 좌표 그리드 (이제 꼭 필요할 때만 계산)
                yy, xx = np.mgrid[y1:y2, x1:x2]
                xx = xx.astype(np.float32)
                yy = yy.astype(np.float32)

                cx_i, cy_i = centers[idx]
                d_cur = (xx - cx_i) ** 2 + (yy - cy_i) ** 2

                for k in uniq:
                    mk = (o_sub == k) & remain
                    if not mk.any():
                        continue
                    cx_k, cy_k = centers[k]
                    d_k = (xx - cx_k) ** 2 + (yy - cy_k) ** 2
                    repl = mk & (d_cur < d_k)
                    if repl.any():
                        c_sub[repl] = s_sub[repl]
                        o_sub[repl] = idx

    return canvas

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


def _dtb_map(h, w):
    """자기 이미지 경계까지의 최소 거리 맵"""
    y = np.arange(h, dtype=np.float32)[:, None]  # (h,1)
    x = np.arange(w, dtype=np.float32)[None, :]  # (1,w)
    # 유효 내부 경계: [l, w-1-r], [t, h-1-b]
    left = x
    right = (w - 1) - x
    top = y
    bottom = (h - 1) - y
    dtb = np.minimum(np.minimum(left, right), np.minimum(top, bottom))
    # 베젤 바깥/경계는 음수가 될 수 있으므로 0으로 클램프
    return np.maximum(dtb, 0.0)  # (h,w) float32


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
        src = _ensure_bgra(img)
        gx, gy = x + shift_x, y + shift_y
        # 클리핑
        x1, y1 = max(0, gx), max(0, gy)
        x2, y2 = min(W, gx + src.shape[1]), min(H, gy + src.shape[0])
        if x1 < x2 and y1 < y2:
            sx1, sy1 = x1 - gx, y1 - gy
            sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
            canvas[y1:y2, x1:x2] = src[sy1:sy2, sx1:sx2]
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
    canvas_dtb = np.full((H, W), -1e9, dtype=np.float32)  # 아주 작은 값으로 초기화

    # 3) 각 이미지를 순회하며 배치
    for img, (x, y) in zip(imgs, positions):
        src = _ensure_bgra(img)
        h, w = src.shape[:2]

        # 이 이미지의 DTB 사전계산
        dtb = _dtb_map(h, w)

        gx, gy = x + shift_x, y + shift_y
        x1, y1 = max(0, gx), max(0, gy)
        x2, y2 = min(W, gx + w), min(H, gy + h)
        if x1 >= x2 or y1 >= y2:
            continue

        sx1, sy1 = x1 - gx, y1 - gy
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

        c_sub = canvas[y1:y2, x1:x2]  # (Hov, Wov, 4)
        d_sub = canvas_dtb[y1:y2, x1:x2]  # (Hov, Wov)
        s_sub = src[sy1:sy2, sx1:sx2]  # (Hov, Wov, 4)
        t_sub = dtb[sy1:sy2, sx1:sx2]  # (Hov, Wov)

        valid = s_sub[:, :, 3] > 0

        # 승자 규칙: 자기 경계에서 더 먼 픽셀이 승리
        take = valid & (t_sub >= d_sub)

        # 동일 색상일 때는 바꿔도/안 바꿔도 결과 동일하니 단순화 가능
        # (원하면 take &= ~same_color 로 덮어쓰기 최소화 가능)
        # same_color = (c_sub[:, :, :3] == s_sub[:, :, :3]).all(axis=2)
        # take = take & ~same_color

        # 갱신
        if take.any():
            c_sub[take] = s_sub[take]
            d_sub[take] = t_sub[take]

    return canvas

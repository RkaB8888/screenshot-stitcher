# screenshot_stitcher\preprocess.py
import numpy as np, cv2


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

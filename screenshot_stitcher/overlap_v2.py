import cv2
import numpy as np


def get_overlap_offset(
    img1,
    img2,
    bezel=(0, 0, 0, 0),
    search_range=2,
    direction="both",
    slack_frac=0.25,
    tol=0,
):
    """
    베젤을 제외한 영역에 제로 패딩과 FFT 위상 상관 기법을 적용하고,
    주변부 픽셀을 검증하는 v2 알고리즘.
    """
    bz_left, bz_top, bz_right, bz_bottom = bezel
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    # 1. 베젤 영역 잘라내기 (각 이미지의 원본 크기 기준)
    crop_h1_end = h1 - bz_bottom if bz_bottom > 0 else h1
    crop_w1_end = w1 - bz_right if bz_right > 0 else w1
    crop1 = img1[bz_top:crop_h1_end, bz_left:crop_w1_end]

    crop_h2_end = h2 - bz_bottom if bz_bottom > 0 else h2
    crop_w2_end = w2 - bz_right if bz_right > 0 else w2
    crop2 = img2[bz_top:crop_h2_end, bz_left:crop_w2_end]

    ch1, cw1 = crop1.shape
    ch2, cw2 = crop2.shape

    # 패딩 길이를 max()가 아니라 두 이미지 길이의 합(+)으로 설정
    pad_h = ch1 + ch2
    pad_w = cw1 + cw2

    # 남는 공간(오른쪽, 아래쪽)을 0으로 채움
    pad1 = cv2.copyMakeBorder(
        crop1, 0, pad_h - ch1, 0, pad_w - cw1, cv2.BORDER_CONSTANT, value=0
    )
    pad2 = cv2.copyMakeBorder(
        crop2, 0, pad_h - ch2, 0, pad_w - cw2, cv2.BORDER_CONSTANT, value=0
    )

    # cv2.phaseCorrelate는 단일 채널의 float32 타입 행렬만 취급함
    float1 = pad1.astype(np.float32)
    float2 = pad2.astype(np.float32)

    # 2. Hanning Window 적용 및 FFT 연산
    # 가장자리의 불연속성 노이즈를 0으로 죽여서 순수 겹침 영역의 피크를 강조
    # window = cv2.createHanningWindow((pad_w, pad_h), cv2.CV_32F)
    # (dx_fft, dy_fft), _ = cv2.phaseCorrelate(float2, float1, window=window)
    (dx_fft, dy_fft), _ = cv2.phaseCorrelate(float2, float1)
    print(f"[INFO] dx_fft: {dx_fft}, dy_fft: {dy_fft}")

    # 3. 소수점 반올림
    fft_dy = int(round(dy_fft))
    fft_dx = int(round(dx_fft))

    # 3.5. 방향(direction)과 흔들림(slack_frac) 허용치를 이용한 가짜 피크 검증
    if direction == "vertical":
        max_slack_x = int(cw1 * slack_frac)
        # 가로 흔들림이 상식선을 넘었다면 FFT가 엉뚱한 패턴에 걸린 것 (매칭 실패)
        if abs(fft_dx) > max_slack_x:
            print(f"[WARN] FFT failed: dx({fft_dx}) exceeds slack({max_slack_x})")
            # fallback=True 로 반환하여 pipeline이 매칭 실패를 인지하게 함
            return 0, int(h1), 0.0, 0.0, 1.0, True

    elif direction == "horizontal":
        max_slack_y = int(ch1 * slack_frac)
        if abs(fft_dy) > max_slack_y:
            print(f"[WARN] FFT failed: dy({fft_dy}) exceeds slack({max_slack_y})")
            return int(w1), 0, 0.0, 0.0, 1.0, True

    # 4. 2D 주변부 정밀 탐색 (검증)
    best_dx, best_dy = fft_dx, fft_dy
    min_diff = float("inf")

    # Y축(세로)과 X축(가로) 주변 ±search_range 픽셀을 모두 탐색
    for offset_y in range(fft_dy - search_range, fft_dy + search_range + 1):
        for offset_x in range(fft_dx - search_range, fft_dx + search_range + 1):

            # 크기가 다른 두 이미지의 정확한 '교집합' 좌표 계산
            x1_start = max(0, offset_x)
            y1_start = max(0, offset_y)
            x1_end = min(cw1, offset_x + cw2)
            y1_end = min(ch1, offset_y + ch2)

            # 겹치는 영역이 아예 없으면 패스
            if x1_start >= x1_end or y1_start >= y1_end:
                continue

            # img2 좌표계 기준 슬라이싱 위치 역산
            x2_start = x1_start - offset_x
            y2_start = y1_start - offset_y
            x2_end = x1_end - offset_x
            y2_end = y1_end - offset_y

            # 겹치는 부분만 슬라이싱
            slice1 = crop1[y1_start:y1_end, x1_start:x1_end]
            slice2 = crop2[y2_start:y2_end, x2_start:x2_end]

            # 두 슬라이스 간의 픽셀 평균 오차 계산 및 tol(허용치) 적용
            diff_matrix = cv2.absdiff(slice1, slice2)
            if tol > 0:
                # tol 이하의 미세한 픽셀 차이는 0으로 무시
                diff_matrix = np.where(diff_matrix <= tol, 0, diff_matrix)
            diff = np.mean(diff_matrix)

            # 가장 오차가 적은 (X, Y) 오프셋을 최종 채택
            if diff < min_diff:
                min_diff = diff
                best_dx = offset_x
                best_dy = offset_y

    # 최종 검증된 오프셋 반환
    return best_dx, best_dy, 1.0, 1.0, 1.0, False

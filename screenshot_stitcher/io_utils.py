# screenshot_stitcher\io_utils.py
import sys, re, cv2, locale
from pathlib import Path
import numpy as np

# OS 로케일(Windows 포함)에 맞춘 문자열 정렬용 설정
try:
    locale.setlocale(locale.LC_ALL, "")
except locale.Error:
    pass  # 실패해도 동작엔 지장 없음

_NUM_RE = re.compile(r"(\d+)")


def _win_like_key(name: str):
    """
    Windows 탐색기의 '이름' 오름차순과 유사한 자연 정렬 키.
    - 숫자 토큰은 int로 비교
    - 문자 토큰은 소문자 처리 후 locale.strxfrm로 변환해 비교
    """
    parts = _NUM_RE.split(name)
    key = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # 숫자
            try:
                key.append(int(part))
            except ValueError:
                key.append(part)  # 혹시 모를 예외 폴백
        else:  # 문자
            key.append(locale.strxfrm(part.lower()))
    return tuple(key)


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

    files.sort(key=lambda x: _win_like_key(x.name))

    paths = [str(f) for f in files]
    print(f"[INFO] 이미지 로드: {len(paths)}장")
    return paths


def _imread_unicode(path: str, flags=cv2.IMREAD_UNCHANGED):
    """한글/비-ASCII 경로에서도 안전한 이미지 로드"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _read_cv_images(paths):
    """경로 리스트 -> cv2 이미지 리스트 (BGRA 유지)"""
    imgs = []
    for p in paths:
        img = _imread_unicode(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[에러] cv2로 이미지 읽기 실패: {p}")
            sys.exit(1)
        imgs.append(img)
    return imgs


def imwrite_unicode(path: str, img) -> bool:
    # 확장자 유추 (없으면 .png)
    ext = Path(path).suffix
    if not ext:
        ext = ".png"
        path = str(Path(path).with_suffix(ext))
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)  # 유니코드 경로 안전
    return True

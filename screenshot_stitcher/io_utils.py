# screenshot_stitcher\io_utils.py
import sys, re, cv2
from pathlib import Path


def _natural_key(s: str):
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

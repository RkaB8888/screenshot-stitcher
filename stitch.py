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


def trim_bezel():
    # TODO: 베젤 인식 및 잘라내기
    return cropped_image


def find_overlap():
    # TODO: 두 이미지 겹침 영역 계산
    return offset


def stitch_images():
    # TODO: 모든 이미지 이어붙이기
    return stitched


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

    for i in range(len(imgs) - 1):
        # 3. 겹침 영역 계산 반복
        dx, dy, conf = find_overlap(imgs[i], imgs[i + 1])
        print(
            f"[OVERLAP] {Path(image_files[i]).name} -> {Path(image_files[i+1]).name} "
            f"dx={dx}, dy={dy}, conf={conf:.3f}"
        )
        # 4. 이어붙인 이미지 베젤 제거

        # 5. 이미지 이어붙이기

    # 6. 출력 저장
    pass


if __name__ == "__main__":
    main()

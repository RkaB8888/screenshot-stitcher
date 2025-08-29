import argparse
import os
import sys


def load_images():
    # TODO: 폴더 내 이미지 읽기
    return images


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
    # 입력 폴더 결정
    if args.input:
        input_dir = os.path.abspath(args.input)
    else:
        # stitch.py가 있는 위치 + images 폴더
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, "images")

    # 폴더 존재 확인
    if not os.path.isdir(input_dir):
        print(f"[에러] 입력 폴더를 찾을 수 없습니다: {input_dir}")
        sys.exit(1)

    print(f"입력 폴더: {input_dir}")
    print(f"출력 파일: {args.output}")

    # 2. 이미지 불러오기
    # 3. 각 이미지 베젤 제거
    # 4. 겹침 영역 계산 반복
    # 5. 이미지 이어붙이기
    # 6. 출력 저장
    pass


if __name__ == "__main__":
    main()

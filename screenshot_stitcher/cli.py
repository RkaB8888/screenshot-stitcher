# screenshot_stitcher\cli.py
import os, time, argparse
from .io_utils import load_images, _read_cv_images
from .pipeline import _accumulate_positions
from .stitch import _stitch_all, _stitch_all_distance
from .progress import _fmt_hms


def parse_args():
    parser = argparse.ArgumentParser(description="Screenshot Stitcher MVP")
    parser.add_argument("--input", type=str, help="입력 이미지 폴더 경로")
    parser.add_argument(
        "--output",
        type=str,
        default="stitched.png",
        help="출력 파일명 (기본: stitched.png)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="vertical",
        choices=["both", "vertical", "horizontal"],
        help="겹침 방향 고정 (both/vertical/horizontal)",
    )
    parser.add_argument(
        "--max-shift-ratio",
        type=float,
        default=1.0,
        help="탐색 범위 비율(각 축별 A/B 크기에 비례). vertical/horizontal에서는 비제한 축에만 주로 영향.",
    )
    parser.add_argument(
        "--tol", type=int, default=5, help="픽셀 차이 허용치(그레이스케일)"
    )
    parser.add_argument(
        "--conf-min", type=float, default=0.05, help="페어 매칭 최소 신뢰도 경고 임계치"
    )
    parser.add_argument(
        "--slack-frac", type=float, default=0.25, help="수직/수평 외축 흔들림 허용 비율"
    )
    parser.add_argument(
        "--no-early-stop",
        dest="early_stop",
        action="store_false",
        help="완전일치(conf=1) 조기 종료 끔",
    )
    parser.add_argument(
        "--early-stop",
        dest="early_stop",
        action="store_true",
        help="완전일치(conf=1) 조기 종료 켬(기본)",
    )
    parser.set_defaults(early_stop=False)

    parser.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.75,
        help="조기 종료 시 요구되는 겹침 면적 비율(0~1)",
    )

    return parser.parse_args()


def main():
    try:
        args = parse_args()
        t0_all = time.perf_counter()

        # main() 입력 폴더 결정부 – CWD 우선, 실패 시 패키지 images 폴백
        if args.input:
            input_dir = os.path.abspath(args.input)
        else:
            cwd_images = os.path.join(os.getcwd(), "images")
            pkg_images = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "images"
            )
            input_dir = cwd_images if os.path.isdir(cwd_images) else pkg_images

        image_files = load_images(input_dir)
        imgs = _read_cv_images(image_files)

        t0_match = time.perf_counter()
        positions, gm, pair_confs = _accumulate_positions(
            imgs,
            direction=args.direction,
            max_shift_ratio=args.max_shift_ratio,
            tol=args.tol,
            conf_min=args.conf_min,
            slack_frac=args.slack_frac,
            early_stop=args.early_stop,
            min_valid_frac=args.min_valid_frac,
        )
        t1_match = time.perf_counter()

        if all(c == 1.0 for c in pair_confs):  # 덮어쓰며 이어 붙이기
            print("[INFO] 완전히 겹침")
            stitched = _stitch_all(imgs, positions)
        else:  # 베젤 제거하며 이어 붙이기
            print("[INFO] 베젤을 고려한 스티칭 시작")
            stitched = _stitch_all_distance(imgs, positions)

        out_path = os.path.abspath(args.output)

        import cv2

        ok = cv2.imwrite(out_path, stitched)
        if not ok:
            print(f"[ERROR] 저장 실패: {out_path}")
        print(f"[OK] saved -> {out_path}  size={stitched.shape[1]}x{stitched.shape[0]}")

        # --- 시간 출력 ---
        total_sec = time.perf_counter() - t0_all
        match_sec = t1_match - t0_match
        print(f"[TIME] match={_fmt_hms(match_sec)}, total={_fmt_hms(total_sec)}")

    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단(Ctrl+C). 중간 진행 상태에서 종료합니다.")

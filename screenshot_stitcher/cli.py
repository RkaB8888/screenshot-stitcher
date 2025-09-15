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
        "--tol", type=int, default=0, help="픽셀 차이 허용치(그레이스케일)"
    )
    parser.add_argument(
        "--conf-min", type=float, default=0.80, help="페어 매칭 최소 신뢰도 경고 임계치"
    )
    parser.add_argument(
        "--slack-frac", type=float, default=0.25, help="수직/수평 외축 흔들림 허용 비율"
    )

    parser.add_argument(
        "--sample-step",
        type=int,
        default=4,
        help="스코어 계산 시 사용할 그리드 샘플 간격(1이면 전체 픽셀 평가)",
    )

    parser.add_argument(
        "--bezel",
        type=str,
        default="0,0,0,0",  # left,top,right,bottom (px)
        help="베젤(무시) 크기: left,top,right,bottom 픽셀 단위. 예) 8,120,8,0",
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
        bz_left, bz_top, bz_right, bz_bottom = map(int, args.bezel.split(","))
        positions, gm, pair_confs, pair_scores, pair_norms = _accumulate_positions(
            imgs,
            direction=args.direction,
            max_shift_ratio=args.max_shift_ratio,
            tol=args.tol,
            conf_min=args.conf_min,
            slack_frac=args.slack_frac,
            sample_step=args.sample_step,
            bezel=(bz_left, bz_top, bz_right, bz_bottom),
        )
        t1_match = time.perf_counter()

        # --- 스티칭 ---
        has_bezel = any(v > 0 for v in (bz_left, bz_top, bz_right, bz_bottom))

        if has_bezel:
            # 베젤이 있으면 반드시 거리 기반으로 처리
            print("[INFO] _stitch_all_distance")
            stitched = _stitch_all_distance(imgs, positions)
        else:
            # 베젤이 없으면 단순 덮어쓰기
            print("[INFO] _stitch_all")
            stitched = _stitch_all(imgs, positions)

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

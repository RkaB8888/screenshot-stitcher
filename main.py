# main.py
import sys
from screenshot_stitcher.cli import main

if __name__ == "__main__":
    # 실행 시 전달된 인자가 파일명(main.py) 딱 하나뿐이라면 = 더블클릭으로 실행했다면
    is_interactive = len(sys.argv) == 1

    if is_interactive:
        print("=" * 50)
        print(" Screenshot Stitcher - 대화형 설정")
        print(" 엔터를 누르면 대괄호 [ ] 안의 기본값이 적용됩니다.")
        print("=" * 50)

        # 1. 입력 폴더
        i_path = input("1. 입력 이미지 폴더 경로 [기본: 현재 폴더의 images/]: ").strip()
        if i_path:
            sys.argv.extend(["--input", i_path])

        # 2. 겹침 방향
        i_dir = input("2. 겹침 방향 (both / vertical / horizontal) [both]: ").strip()
        if i_dir in ["both", "vertical", "horizontal"]:
            sys.argv.extend(["--direction", i_dir])

        # 3. 베젤 설정
        i_bezel = input("3. 베젤 크기 (좌,상,우,하 픽셀) [10,10,10,10]: ").strip()
        if i_bezel:
            sys.argv.extend(["--bezel", i_bezel])

        # 4. 알고리즘 선택
        i_method = input("4. 알고리즘 선택 (v1 / v2) [v2]: ").strip()
        if i_method in ["v1", "v2"]:
            sys.argv.extend(["--method", i_method])

        if i_method == "v1":
            # 3. 샘플링 설정
            i_sample = input("5. 샘플링 간격 선택 [8]: ").strip()
            if i_sample:
                sys.argv.extend(["--sample-step", i_sample])

        print("-" * 50)
        print("[INFO] 스티칭을 시작합니다...\n")

    try:
        # sys.argv에 인자들을 채워 넣었으니, 이제 cli.main()을 호출하면 알아서 파싱됨!
        main()
    except Exception as e:
        print(f"\n[ERROR] 스크립트 실행 중 오류가 발생했습니다: {e}")
    finally:
        # 더블클릭(대화형)으로 실행했을 때만 창이 안 닫히고 대기하도록 처리
        if is_interactive:
            input("\n[완료] 창을 닫으려면 엔터 키를 누르세요...")

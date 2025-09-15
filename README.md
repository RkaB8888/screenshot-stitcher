# screenshot-stitcher

여러 장의 스크린샷을 겹치는 영역을 자동 인식하여 이어붙이는 CLI 도구.

## 목표

* 스크린샷 간 겹침 인식 및 정합
* 베젤 제거 및 블렌딩(향후 예정)

## 현재 상태 (MVP)

* 전제: 동일 배율(줌), 같은 원본 소스, 회전/반전/원근 왜곡 없음

* 기능:

  * 입력 폴더 내 PNG 이미지 자동 로드 및 정렬
  * 회색조 + 알파마스크 변환 및 캐싱
  * 브루트포스 기반 겹침 탐색 (`find_overlap_gray`)

    * tol(허용 오차), conf-min(신뢰도 기준), slack-frac(허용 흔들림) 옵션 지원
    * direction 옵션 (both/vertical/horizontal) 지원
    * **최적화:** 겹침 면적 내림차순 탐색 및 샘플링(`--sample-step`) 지원
  * 진행률 표시 및 매칭 로그 출력
  * **스티칭 분기 로직**

    * 모든 베젤 값이 0이면 → 단순 덮어쓰기(`_stitch_all`)
    * 하나라도 베젤 > 0이면 → 거리 기반 안티-베젤 스티칭(`_stitch_all_distance`)
  * 겹친 이미지를 이어붙여 PNG 저장 (`stitched.png`)

* CLI 옵션:

  * `--input`: 입력 폴더 경로
  * `--output`: 출력 파일명
  * `--direction`: 겹침 방향
  * `--max-shift-ratio`: 탐색 범위 비율
  * `--tol`: 픽셀 차이 허용치
  * `--conf-min`: 신뢰도 임계값
  * `--slack-frac`: 흔들림 허용 비율
  * `--sample-step`: 매칭 시 샘플링 간격(기본 4, 1이면 전체 픽셀 평가)
  * `--bezel`: 베젤 크기(left,top,right,bottom, 기본 `0,0,0,0`)

## 코드 구조

리팩터링을 통해 모듈 단위로 분리되어 있습니다:

```
screenshot_stitcher/
 ├─ cli.py            # CLI 인자 파싱 및 실행 흐름
 ├─ io_utils.py       # 이미지 로드 관련 함수
 ├─ preprocess.py     # 그레이 변환 및 마스크 처리
 ├─ overlap.py        # 겹침 탐색 로직
 ├─ pipeline.py       # 이미지 정합 및 좌표 누적
 ├─ stitch.py         # 스티칭 및 DTB 기반 승자 규칙
 └─ progress.py       # 진행률/시간 표시
```

실행은 `main.py`를 통해 이루어집니다:

```bash
python main.py --input ./images --output stitched.png
```

## 로드맵

### 점진 업그레이드

* 탐색 속도 최적화 (phase correlation, matchTemplate, integral image 등)
* 겹침 정도(20\~40% 등)에 따른 정합 로직 개선
* 실패 케이스 원인 및 가이드 메시지 강화
* 베젤 자동 감지/제거 옵션 추가

### GUI 지원

* 드래그 앤 드롭으로 이미지 순서/방향 지정
* 회전/뒤집기 사전 조치 UI
* 겹침 미리보기, 실패 구간 하이라이트
* 파라미터 조절 슬라이더 (겹침 임계값, 블렌딩 강도 등)

### 탐구

* 다양한 변환 보정(아핀/호모그래피) 고도화
* 시임 찾기 및 멀티밴드 블렌딩
* 동영상 입력(프레임 추출 후 스티칭) 가능성 검토
* 대용량 처리(피라미드/타일링) 안정화

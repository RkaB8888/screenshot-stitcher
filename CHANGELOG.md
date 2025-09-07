# Changelog

## \[Unreleased]

* TBD

## 2025-09-07

### Added

* `--early-stop` / `--no-early-stop` CLI 옵션 추가

  * conf==1.0 && 충분한 겹침 면적일 때 조기 종료
* `--min-valid-frac` CLI 옵션 추가

  * 조기 종료 시 요구되는 최소 겹침 비율 제어 가능

### Changed

* 겹침 후보를 단순 반복 → **겹침 면적 내림차순 탐색**으로 최적화
* 대규모 이미지 스티칭 시 탐색 성능 개선 (29장 기준 약 11분 단축)

### Fixed

* 겹침 영역이 없는 경우 후보 리스트 비어있을 때 안전하게 0 반환

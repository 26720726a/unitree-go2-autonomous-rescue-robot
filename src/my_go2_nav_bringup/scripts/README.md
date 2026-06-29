# my_go2_nav_bringup / scripts

Go2 보행 브리지 및 유틸리티 스크립트. `setup.py` 의 `glob('scripts/*.py')` 로
`share/my_go2_nav_bringup/scripts/` 에 설치된다. (`_archive/` 하위는 비재귀 glob 라 설치 안 됨)

## 현재 사용 파일

| 파일 | 용도 | 비고 |
|------|------|------|
| `cmd_vel_to_sport_demo.py` | **표준 cmd_vel→sport 브리지.** AI 모드 전환 후 ClassicWalk 보행 + rotate-to-face + slew + watchdog | v0~v6 실험본을 통합한 정리판. 신규 작업은 이 파일 기준 |
| `cmd_vel_to_sport.py`      | 구 표준 브리지 (AI 모드 전환 없음) | `go2_mapping_utlidar.launch.py` 가 참조 중 |
| `cmd_vel_to_sport_gpt.py`  | `unitree_sdk2py.SportClient` 사용 버전 | `go2_mapping.launch.py` 가 참조 중. SDK 설치 필요 |
| `classic_walk_demo.py`     | ClassicWalk 단독 데모 (전진/회전 N초) | `--vx/--vyaw/--duration/--speed` |
| `check_motion_mode.py`     | motion_switcher 현재 모드 조회 (CheckMode 1001) | 디버그용 |
| `test_gait.py`             | gait 전환 실험 | 디버그용 |
| `restamp_cloud.py`         | 라이다 포인트클라우드 타임스탬프 재기록 | launch 가 참조 |

## ClassicWalk 핵심

`ClassicWalk(api_id=2049)` 는 Go2 의 **AI(고급) sport 전용 gait** 다. 기본(normal)
sport 모드에서 보내면 무시되므로, 먼저 `motion_switcher` 로 `SelectMode("ai")`
(api_id=1002, `{"name":"ai"}`) 를 보낸 뒤 ClassicWalk 를 켜야 한다.
`cmd_vel_to_sport_demo.py` 가 이 순서를 자동 처리한다.

부팅 시퀀스: `SelectMode("ai") → BalanceStand → SpeedLevel → ClassicWalk(True) → /cmd_vel 수락`

## 실행

```bash
source ~/capstone/install/setup.bash
python3 cmd_vel_to_sport_demo.py
# 파라미터 예
python3 cmd_vel_to_sport_demo.py --ros-args -p speed_level:=fast -p set_classic_walk:=true
```

## `_archive/`

지난 실험본/백업(`*_v0~v6`, `*.bak`, `cmd_vel_classic_walk.cpp`). 삭제하지 않고 보관만 함.
`cmd_vel_to_sport_demo.py` 가 이들을 대체한다.

# JitterAnalysisTool_Yokogawa_ver

Seismic project에서 Jitter 측정 데이터를 분석하는 툴
Yokogawa DLM4000 오실로스코프로 측정한 데이터를 분석해줌

## 사용법
1. Data 폴더에 측정된 CSV 파일들을 넣는다.
2. python main.py를 실행한다.
3. output 폴더에 결과가 저장된다.

## 지원 오실로스코프
- Yokogawa DLM4000 (DSL4000) 시리즈

## 분석 기능
- CH1과 CH2 사이의 지연 시간(Delay) 측정
- 동적 Midpoint 트리거링 (50% 전압 임계값 자동 설정)
- 상승/하강 에지 자동 감지
- 파일별 파형 플롯 생성
- 전체 Delay 추세 및 분포 통계 요약

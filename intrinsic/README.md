# Intrinsic Evaluation

KorPress의 최대 span 길이 `L`, `drop_rule`, `threshold`가 압축률과 정보 보존에 미치는 영향을 평가합니다.

## 주요 스크립트

| 파일 | 역할 |
|---|---|
| `prepare_aihub_input.py` | AI Hub 발화 데이터 정리 |
| `build_span_candidates.py` | Stanza 기반 span 후보 생성 |
| `prepare_span_data.py` | 학습 데이터 생성 |
| `train_span_encoder.py` | span 분류기 학습 |
| `predict_span_encoder.py` | span DROP 확률 예측 |
| `run_intrinsic_experiment.py` | 전체 조건 평가 |

## 실험 조건

| 변수 | 값 |
|---|---|
| `L` | 1, 2, 4, 8 |
| `drop_rule` | max, mean, min |
| `threshold` | 0.5, 0.7, 0.9 |

결과는 `results/intrinsic_summary.csv`에 저장합니다.

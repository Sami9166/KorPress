# QA Task

KorQuAD 2.0 문맥을 Token 및 Span 방식으로 압축하고 Qwen3-8B의 QA 성능을 비교합니다.

## 실험 설정

- 데이터: KorQuAD 2.0, 694개 QA
- Reader: Qwen3-8B
- 예산: retention rate 0.9, 0.8, 0.7
- Span: `L=1,2,4,8`, drop rule `max, mean, min`
- 평가지표: EM, F1
- Token baseline: 동일한 retention rate 적용

QA에서는 threshold 대신 목표 reader-token retention rate를 기준으로 압축합니다.

## 실행

실행 인자는 다음 명령으로 확인합니다.

```bash
python qa_task/run_korquad_qa_experiment.py --help
```

## 결과

결과 파일: `results/qa_task.csv`

현재 CSV는 `source=Drive_prediction_aggregate`, `official_evaluator=False`로 기록되어 있습니다. 공식 evaluator 결과를 확보하면 해당 메타데이터와 수치를 함께 갱신합니다.

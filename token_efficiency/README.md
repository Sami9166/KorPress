# Token Efficiency

Belebele 영한 병렬 문항에서 LLMLingua-2 압축의 token 효율과 정보 전달을 비교합니다. 이 실험은 KorPress의 한국어 압축 연구 동기를 확인하기 위한 분석입니다.

## 실험 설정

- 데이터: Belebele test split, 언어별 300개 문항
- Compressor: LLMLingua-2
- 압축률: 0.33, 0.50, 0.75
- 평가 모델: Claude Haiku
- Token 기준: XLM-R tokenizer와 `o200k_base`

## 실행

```bash
python token_efficiency/run_token_efficiency.py self-check
python token_efficiency/run_token_efficiency.py sample
python token_efficiency/run_token_efficiency.py compress
python token_efficiency/run_token_efficiency.py evidence
python token_efficiency/run_token_efficiency.py evaluate
python token_efficiency/run_token_efficiency.py metrics
python token_efficiency/run_token_efficiency.py plot
```

## 결과 파일

| 파일 | 내용 |
|---|---|
| `results/token_metrics.csv` | token 수와 보존율 |
| `results/accuracy.csv` | 전체 정확도 |
| `results/accuracy_hard.csv` | passage-dependent 정확도 |
| `results/transfer_efficiency.csv` | 정보 전달 효율 |
| `results/evidence_survival.csv` | 정답 근거 보존율 |
| `results/evidence_per_item.csv` | 문항별 근거 결과 |
| `results/matched_budget.csv` | 유사 token 예산 비교 |

중간 JSONL 파일은 `data/`, 그림은 `figs/`에 저장합니다.

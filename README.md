# KorPress

한국어 dependency span 압축기와 token baseline의 학습·intrinsic·KorQuAD QA 실험 코드입니다.

## 디렉터리

```text
src/
  dependency_spans.py          # Stanza 파싱 + dependency span 후보 생성
  compressor.py                # L/threshold/drop-rule 압축 로직
  span_encoder_data.py         # span encoder JSONL·문맥 window·정렬
  span_encoder_model.py        # KLUE-RoBERTa contextual span classifier
  evaluate_compression.py      # 수치·날짜·부정어·문법 보존 지표
  experiment_runtime.py        # intrinsic/QA 공통 로더·Qwen3 설정·adapter
  token_baseline.py            # token-level baseline
script/
  prepare_aihub_input.py       # AI Hub chunks.csv → 문장/metadata
  build_span_candidates.py     # 파싱 + L별 span 후보 생성
  check_integrity.py           # 후보 JSON 전수 구조 검사
  print_readable_trees.py      # dependency tree 샘플 확인
  prepare_span_data.py         # span_labels.csv.gz → train/val/test JSONL
  verify_span_data.py          # 준비 데이터 개수·라벨 검증
  train_span_encoder.py        # span encoder 학습·resume
  predict_span_encoder.py      # encoder DROP 확률 생성
  run_grid.py                  # 선택 진단: 문법·고유명사·압축 latency grid
  run_intrinsic_experiment.py  # Span hyperparameter intrinsic sweep
  run_korquad_qa_experiment.py # calibration 후 Span/Token QA 재평가
  prepare_subword_teacher_input.py # KLUE tokenizer 기반 ChatGPT 입력 생성
  convert_subword_teacher_output.py # teacher keep_indices → subword label CSV
  token_baseline.py            # subword token baseline 학습·예측 CLI
```

## 데이터

AI Hub 원본 `chunks.csv`, span 라벨, prepared JSONL, 모델 가중치는 저장소에 포함하지 않습니다.
실행 시 별도로 준비하고, `chunks.csv`의 `sentence_id`가 span records/predictions와 일치해야 합니다.
현재 `data/korquad/chunks.csv`는 KorQuAD용 샘플입니다.

## 기본 흐름

```powershell
python -m pip install -r requirements.txt
python script/prepare_aihub_input.py --input chunks.csv --out-dir prepared/input
python script/build_span_candidates.py --input prepared/input/sentences.txt --metadata prepared/input/metadata.csv --output prepared/span_candidates.json
python script/prepare_span_data.py --chunks chunks.csv --spans span_labels.csv.gz --output-dir prepared/aihub
python script/train_span_encoder.py --data-dir prepared/aihub --chunks chunks.csv --output-dir runs/span_encoder --model-name klue/roberta-base --pooling mean_max --precision fp16
python script/predict_span_encoder.py --checkpoint runs/span_encoder/best_model --data prepared/aihub/test.jsonl --chunks chunks.csv --output runs/span_encoder/test_predictions.csv.gz
python script/run_intrinsic_experiment.py --chunks chunks.csv --span-records prepared/aihub/validation.jsonl --span-predictions runs/span_encoder/validation_predictions.csv.gz --output-dir experiments/intrinsic_qwen3_8b_validation --qwen-model Qwen/Qwen3-8B --drop-rules max mean min
python script/run_korquad_qa_experiment.py --chunks data/korquad/chunks.csv --qa-pairs data/korquad/qa_pairs.json --span-records data/korquad/spans.jsonl --span-predictions data/korquad/korquad_100_predictions_csv.gz --calibration-chunks data/korquad/chunks.csv --calibration-span-records data/korquad/spans.jsonl --calibration-span-predictions data/korquad/korquad_100_predictions_csv.gz --token-checkpoint runs/token_klue_roberta_base/best_model --output-dir experiments/qa_qwen3_8b --qwen-model Qwen/Qwen3-8B --torch-dtype float16
python script/prepare_subword_teacher_input.py --chunks chunks.csv --tokenizer klue/roberta-base --tokenizer-path encoder/klue_roberta_base_mean_max/best_model/tokenizer --output data/subword_teacher_input.jsonl
python script/convert_subword_teacher_output.py --teacher-input data/subword_teacher_input.jsonl --teacher-output data/subword_teacher_output.jsonl --output data/subword_labels.csv.gz
python script/token_baseline.py train --chunks chunks.csv --labels data/subword_labels.csv.gz --output-dir runs/token_klue_roberta_base --model-name klue/roberta-base --tokenizer-name encoder/klue_roberta_base_mean_max/best_model/tokenizer --max-length 512 --batch-size 8 --epochs 3 --fp16 --device cuda
```

`run_intrinsic_experiment.py`는 Token을 비교하지 않고 Span의 `L`, `drop_rule`,
`threshold` 영향만 전수 평가합니다. Qwen 모델 기본값은 `Qwen/Qwen3-8B`입니다.
기본 Span grid는 `drop_rule={max,mean,min}`, `L={1,2,4,8}`,
`threshold={0.5,0.7,0.9}`(36개)입니다. `intrinsic_summary.csv`가 전체 결과이고
`intrinsic_selected_settings.csv`는 목표 삭제율별 대표 Span 설정입니다.

QA는 calibration 문맥에서 전달된 **전체 Span/Token 후보 조합**을 실제 Qwen tokenizer
삭제율로 비교합니다. 목표 삭제율(`--targets`, 기본 10/20/30%)별로 가장 가까우면서
Span–Token CR 차이가 작은 조합을 선택하고, 선택된 설정만 QA 전체 문맥에서 reader로
평가합니다. 선택 결과는 `qa_selected_settings.csv`, 전체 조합 진단은
`qa_calibration_grid.csv`에 저장됩니다. `--all-settings`를 주면 calibration 선택을
건너뛰고 모든 평가 후보를 실행합니다. 별도 calibration 경로를 생략하면 QA 문맥을
fallback으로 사용하므로 최종 실험에서는 별도 문맥을 지정해야 합니다. QA는 원문 조건을
먼저 실행해 `qa_original_summary.json`에 기준 EM/F1과
원문 reader latency를 저장하고, 압축 조건에는 절대·상대 손실(`em_loss`, `f1_loss` 등)과
압축 문맥 reader latency를 기록합니다. 질문별 latency는
`qa_results_*.json`의 `reader_latency_s_original`/`reader_latency_s_compressed`에,
집계값(mean/median/p95/total)은 `qa_summary.csv`에 저장됩니다. 원문 latency 원자료는
`qa_original_latency.json`에 별도로 저장됩니다. Latency는 모델 로딩과 압축 후보 생성 시간을
제외한 Qwen reader 생성 시간이며, CUDA에서는 전후 synchronize 후 측정합니다. 모델을
바꾸려면 두 명령 모두 `--qwen-model` 하나만 지정합니다(`--qwen-tokenizer`, `--qa-model`은
호환용 별칭).

Intrinsic의 주 비교 표 항목은 목표 삭제율, method/rule/L/threshold, 실제 Qwen-token
삭제율과 CR 차이, 숫자·날짜·부정 단서 보존율, 의미 유사도입니다. 의미 유사도는
기본적으로 `dragonkue/BGE-m3-ko`의 임베딩 코사인 유사도이며, 필요하면
`--similarity-model none`으로 끌 수 있습니다. `semantic_mean`뿐 아니라
`semantic_p10`과 `semantic_min`도 `intrinsic_summary.csv`에 남아 최악의 보존 사례를
확인할 수 있습니다.

Token teacher 출력은 `subword_labels.csv.gz`로 변환한 뒤 subword 단위로 학습합니다.
라벨 생성·학습·추론에 같은 KLUE tokenizer를 사용해야 합니다.
추론에서도 각 subword의 DROP 확률을 threshold와 비교하고, KEEP한 subword ID만
tokenizer로 decode합니다. Span과 동일하게 각 utterance를 포함하는 512-subword
window를 구성하고, 겹치는 window의 점수는 평균냅니다. 일반적인 sliding-window는
사용하지 않으며, utterance 자체가 window보다 길면 오류를 냅니다.

Qwen3는 `transformers<4.51`에서 로드되지 않으므로 `requirements.txt`의 최소 버전을
그대로 사용해야 합니다. QA는 비교 가능성을 위해 chat template의 thinking을 끄고
greedy decoding을 사용합니다.

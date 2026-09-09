# Project2 Experiment Notes

이 문서는 `C:\myproject2026_2\outputs` 안에 있는 실험 결과가 각각 어떤 설정으로 나온 것인지 빠르게 기억하기 위한 정리다.

## 공통 설정

| 항목 | 설정 |
|---|---|
| Task | 5-class modulation classification |
| Modulations | BPSK, QPSK, 8PSK, 16QAM, 64QAM |
| Input length | 4096 I/Q samples |
| Input shape | `(2, 4096)` |
| Channel/noise | synthetic AWGN SNR sweep |
| Test samples | 보통 SNR별 modulation당 500개, 즉 SNR당 2,500 samples |
| Class chance level | 20% |
| Main models | SL baseline, contrastive SSL pretrain + fine-tune |
| Main low-label ratios | 0.0025, 0.005, 0.01 |
| Additional original ratios | 0.1, 0.2, 1.0 for SL; 0.1, 0.2 for SSL |
| Fine-tune/train epochs | 대부분 200 epochs |
| Batch size | 대부분 16 |
| Optimizer settings | encoder lr `1e-5`, classifier lr `2e-4`, weight decay `1e-4` |
| SSL pretrained encoder | contrastive SSL encoder, usually epoch 20 checkpoint for fine-tuning |

주의: `label_ratio`는 전체 synthetic train split에서 labeled subset을 얼마나 쓰는지에 해당한다. 결과 json의 `num_per_class=200`은 스크립트 내부 기록값으로 남아있지만, 실제 labeled sample 수는 ratio에 따라 달라진다.

## 현재 summary 파일

| 파일 | 내용 |
|---|---|
| `outputs/summary/summary.xlsx` | 현재 통합 엑셀 요약 |
| `outputs/summary/experiment_notes.md` | 이 문서 |

`summary.xlsx`의 주요 시트:

| Sheet | 내용 |
|---|---|
| `Sheet1` | 기존 `5type_l4096_snr0to20` 모델을 dense SNR test로 재평가한 결과 |
| `SL_SSL_Compare` | 같은 label ratio와 SNR에서 SL vs SSL 직접 비교 |
| `SNR별 비교` | SNR을 행으로 두고 ratio별 SL/SSL/gain을 옆으로 펼친 표 |
| `LowSNR Detail` | low-SNR train/fine-tune exposure 실험 상세 |
| `LowSNR Pivot` | low-SNR exposure 실험의 SNR별 wide table |
| `DenseMethods Detail` | dense SNR, SNR-conditioned, denoising 실험 상세 |
| `DenseMethods Pivot` | dense method 실험의 SNR별 wide table |
| `Overview` | low/transition/high SNR 평균 요약 |

---

## 1. `outputs/5type_l4096_snr0to20`

원래 기준 실험 폴더다. 학습 SNR은 `0,5,10,15,20`이고, 기존 test SNR은 `-10,-5,0,5,10,15,20`이다.

### 목적

기본 SL과 contrastive SSL의 label efficiency를 비교하기 위한 baseline.

### 구성

| 하위 폴더 | 의미 |
|---|---|
| `sl/SL_*` | supervised baseline. random/init encoder + classifier를 labeled subset으로 학습 |
| `ssl_pretrain/` | contrastive SSL pretraining 결과 |
| `ssl_eval/SSL_*` | SSL pretrained encoder를 불러와 labeled subset으로 fine-tune |
| `trial_1_0p005_only/` | 0.005 ratio만 별도로 다시 돌린 trial |

### 완료된 주요 ratio

| Method | Ratios |
|---|---|
| SL | 0.0025, 0.005, 0.01, 0.1, 0.2, 1.0 |
| SSL | 0.0025, 0.005, 0.01, 0.1, 0.2 |

### 대표 설정

| 항목 | 값 |
|---|---|
| Train SNR | `0,5,10,15,20` |
| Original test SNR | `-10,-5,0,5,10,15,20` |
| Epochs | 200 |
| Batch size | 16 |
| SSL fine-tune pretrained path | `ssl_encoder_cuda_epoch_20.pth` 계열 |

### 읽는 법

각 ratio 폴더 안에서:

| 파일 | 의미 |
|---|---|
| `*_results.csv` | SNR별 accuracy 및 modulation별 accuracy |
| `*_confusion.csv` | SNR별 confusion matrix long format |
| `*_summary.json` | best val acc, checkpoint, 설정 |
| `*_metrics.json` | SNR별 acc/confusion matrix JSON |
| `*_train_log.csv` | epoch별 train/val 기록 |

---

## 2. `outputs/dense_snr_eval_existing_drive_ep200`

`5type_l4096_snr0to20_runtime_data_pretrained_zip_ep200`에 있던 기존 체크포인트를 새 dense SNR test set으로 eval-only 재평가한 결과다.

### 목적

기존 모델을 다시 학습하지 않고, `1,2,3,4 dB` 구간을 추가해서 성능 전이 구간을 촘촘히 확인.

### 설정

| 항목 | 값 |
|---|---|
| Training | 없음. eval-only |
| Checkpoints | 기존 SL/SSL best checkpoints |
| Test SNR | `-10,-5,0,1,2,3,4,5,10,15,20` |
| Test samples | SNR당 2,500 samples |
| Output | 새 결과만 별도 저장, 원래 결과 덮어쓰기 없음 |

### 완료된 ratio

| Method | Ratios |
|---|---|
| SL | 0.0025, 0.005, 0.01, 0.1, 0.2, 1.0 |
| SSL | 0.0025, 0.005, 0.01, 0.1, 0.2 |

### 핵심 해석

`-10,-5 dB`에서는 거의 chance level 근처다. `0 dB`부터 SSL이 SL보다 조금 나아지기 시작하고, `1~5 dB` 사이에서 성능 상승 곡선이 급격히 나타난다. 그래서 이 폴더는 “low SNR 전체가 아니라 transition SNR이 어디인지”를 보여주는 데 중요하다.

---

## 3. `outputs/5type_l4096_lowlabel_low_snr_exposure`

low-SNR을 train/pretrain/fine-tune에 포함했을 때 정말 저 SNR 성능이 올라가는지 확인하기 위한 실험 묶음이다.

### 목적

단순히 SSL이 좋다는 주장 대신, **low-SNR exposure를 어디에 넣어야 의미가 있는지** 비교하기 위한 설계.

### 기본 비교 구조

| 실험 | SL train SNR | SSL pretrain SNR | SSL fine-tune SNR | 비교 목적 |
|---|---:|---:|---:|---|
| A `current_sl` | `0,5,10,15,20` | - | - | 기존 SL 기준 |
| B `current_ssl` | - | `0,5,10,15,20` | `0,5,10,15,20` | 기존 SSL 기준 |
| C `low_snr_sl` | `-10,-5,0,5,10,15,20` | - | - | low-SNR labeled training 효과 |
| D `low_pretrain_ssl` | - | `-10,-5,0,5,10,15,20` | `0,5,10,15,20` | SSL pretrain에 low-SNR 노출 |
| E `low_finetune_ssl` | - | `0,5,10,15,20` | `-10,-5,0,5,10,15,20` | SSL fine-tune에 low-SNR labeled exposure |
| F `low_both_ssl` | - | `-10,-5,0,5,10,15,20` | `-10,-5,0,5,10,15,20` | pretrain/fine-tune 모두 low-SNR 노출 |

### 현재 완료된 결과

| Condition | 완료 ratio |
|---|---|
| A `current_sl` | 0.0025, 0.005, 0.01 |
| B `current_ssl` | 0.0025, 0.005, 0.01 |
| C `low_snr_sl` | 0.0025, 0.005, 0.01 |
| D `low_pretrain_ssl` | 0.0025, 0.005 |
| E `low_finetune_ssl` | 0.005 |
| F `low_both_ssl` | 0.01 |

### Test SNR

`-10,-5,0,5,10,15,20`

### 핵심 해석

low-SNR을 train/fine-tune에 추가해도 `-10,-5 dB`는 거의 개선되지 않았다. 즉 단순 exposure로는 severe low SNR을 극복하기 어렵고, 실제 의미 있는 구간은 `0~5 dB` 또는 그 이상이다. 이 실험은 “왜 low SNR이 안 오르는지”를 논문에서 객관적으로 말할 수 있는 근거다.

---

## 4. `outputs/5type_l4096_dense_snr_denoising_snrcond`

dense SNR test, SNR-conditioned classifier, denoising pretraining을 비교한 실험 묶음이다.

### 목적

`-10,-5,0`이 정말 극복 불가능한지, 아니면 방법을 바꾸면 좋아지는지 확인.

### Test SNR

`-10,-5,0,1,2,3,4,5,10,15,20`

### 실험 조건

| Condition | 의미 |
|---|---|
| A `dense_sl_single_head` | dense train/test SNR에서 SL single-head baseline |
| B `dense_contrastive_ssl_single_head` | 기존 contrastive SSL + single-head fine-tune |
| C `dense_ssl_snr_conditioned` | SSL encoder + SNR-conditioned classifier |
| D `dense_denoising_ssl_single_head` | MSE denoising SSL pretrain + single-head fine-tune |
| `denoising_pretrain` | denoising pretraining checkpoint/log |

### 현재 완료된 결과

| Condition | 완료 ratio |
|---|---|
| A dense SL | 0.0025, 0.005 |
| B contrastive SSL | 0.0025 |
| C SNR-conditioned SSL | 0.005 결과 있음, 0.0025는 summary/checkpoint 일부만 있음 |
| D denoising SSL | 0.0025, 0.005 |

주의: 이 폴더는 일부 ratio가 부분적으로만 완료되어 있다. `summary.xlsx`는 실제 `*_results.csv`가 있는 것만 반영한다.

### 핵심 해석

| 방법 | 관찰 |
|---|---|
| Contrastive SSL | 0 dB 이상에서 SL보다 뚜렷하게 좋고, 1~5 dB transition 구간에서 상승이 큼 |
| SNR-conditioned classifier | `2~3 dB` 근처에서 일부 개선이 있었지만 `-10,-5,0 dB`를 본질적으로 해결하지는 못함 |
| MSE denoising SSL | 현재 구현/설정에서는 거의 chance level로 실패. denoising 전체가 실패라는 뜻은 아니고, “단순 MSE reconstruction objective는 분류 표현에 안 맞았다”는 negative result로 해석 |
| SL | 저라벨에서는 특히 mid/high SNR에서 SSL보다 약함 |

---

## 현재까지의 큰 결론

1. `-10,-5 dB`는 어떤 단순 방법도 안정적으로 개선하지 못했다.
2. `0 dB`는 애매한 경계다. contrastive SSL은 SL보다 나아지지만, 완전히 해결된 수준은 아니다.
3. 핵심 transition은 `1~5 dB`, 특히 `2~4 dB` 근처에 있다.
4. low-SNR exposure를 추가해도 severe low SNR이 자동으로 해결되지는 않는다.
5. SNR-conditioned classifier는 “일부 transition 구간 개선” 정도로 볼 수 있다.
6. MSE denoising pretraining은 현재 결과상 실패한 방법이다.
7. 논문/포스터에서는 “SSL이 무조건 좋다”보다 **SNR regime별로 SSL gain이 다르게 나타난다**는 방향이 더 설득력 있다.

## 논문에서 쓰기 좋은 framing

추천 메시지:

> Contrastive SSL improves modulation classification mainly in the low-label and moderate-to-high SNR regimes, while severe low SNR remains near the information/decision boundary for this synthetic setup. Dense SNR evaluation reveals that the practical transition occurs around 1-5 dB, rather than uniformly across all low SNR values.

한국어로는:

> SSL은 모든 저 SNR을 해결하는 만능 방법이라기보다, 라벨이 적고 SNR이 어느 정도 확보되는 전이 구간에서 성능 이득이 크게 나타난다. 특히 -10, -5 dB에서는 대부분의 방법이 chance level에 머물렀고, 1~5 dB 구간에서 방법 간 차이가 본격적으로 드러났다.

## 앞으로 추가하면 좋은 최소 실험

| 우선순위 | 실험 | 이유 |
|---|---|---|
| 1 | dense SNR 결과 시각화 정리 | 이미 결과가 있고 논문 그림으로 바로 사용 가능 |
| 2 | modulation별 per-SNR accuracy/confusion 정리 | 교수님 지적처럼 high-order QAM이 low SNR에서 의미 있는지 설명 가능 |
| 3 | 현실적 SNR mask 적용 분석 | 64QAM/16QAM을 severe low SNR에서 평가하는 것의 한계를 명확히 제시 가능 |
| 4 | seed 1개 추가 반복 | random artifact인지 줄이는 근거 |

## 파일 위치 빠른 참조

| 목적 | 경로 |
|---|---|
| 통합 엑셀 요약 | `outputs/summary/summary.xlsx` |
| 원래 baseline 결과 | `outputs/5type_l4096_snr0to20` |
| 기존 checkpoint dense re-eval | `outputs/dense_snr_eval_existing_drive_ep200` |
| low-SNR exposure 결과 | `outputs/5type_l4096_lowlabel_low_snr_exposure` |
| dense/SNR-conditioned/denoising 결과 | `outputs/5type_l4096_dense_snr_denoising_snrcond` |
| 관련 figures | 각 결과 폴더의 `figures/` |

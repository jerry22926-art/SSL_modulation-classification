# SNR 변화 환경에서의 자기지도 변조 분류

이 폴더는 저라벨 변조 분류 실험에 사용한 코드, 논문용 그림, 핵심 결과 JSON을 정리한 배포용 묶음입니다.

논문 링크용으로 가볍게 보기 좋게 구성했으며, 큰 데이터 파일, 모델 체크포인트, Colab 노트북, CSV 덤프는 제외했습니다.

## 실험 설정

| 항목 | 설정 |
|---|---|
| 과제 | 5-class modulation classification |
| 변조 방식 | BPSK, QPSK, 8PSK, 16QAM, 64QAM |
| 입력 | I/Q 신호, shape `(2, 4096)` |
| 채널 | Synthetic AWGN SNR sweep |
| 기본 train SNR | `0, 5, 10, 15, 20` dB |
| Low-SNR exposure SNR | 기본 train SNR에 `-10, -5` dB 추가 |
| Per-head SSL train group | severe `[-10, -5]`, transition `[0, 5]`, high `[10, 15, 20]` |
| 주요 label ratio | `0.005` |
| labeled training 규모 | baseline `0.005`는 modulation class당 약 200개 labeled sample 사용; per-head SSL은 head당 200개, 총 600개 사용 |
| Dense test SNR | `-10, -5, 0, 1, 2, 3, 4, 5, 10, 15, 20` dB |
| 테스트 샘플 수 | SNR별 modulation당 500개, SNR별 총 2500개 |
| Chance level | 20% |
| 주요 방법 | SL, contrastive SSL, low-SNR exposure SSL, per-head SSL, denoising SSL |

## 모델 구조와 학습 방법

### Backbone

주요 SL/SSL 실험은 동일한 1D residual CNN backbone을 사용합니다.

```text
Input I/Q signal
    shape: (2, 4096)
        |
        v
Sample별 std normalization
        |
        v
ResBlock 1D: 2 -> 64    + MaxPool1d(2)
        |
        v
ResBlock 1D: 64 -> 128  + MaxPool1d(2)
        |
        v
ResBlock 1D: 128 -> 256 + MaxPool1d(2)
        |
        v
ResBlock 1D: 256 -> 512 + MaxPool1d(2)
        |
        v
Flattened encoder feature
```

- 입력: I/Q 2채널, sequence length 4096.
- Encoder: residual 1D convolution block 4개.
- 채널 수: `2 -> 64 -> 128 -> 256 -> 512`.
- Downsampling: 각 residual block 뒤에 max pooling 적용.
- Feature vector: 마지막 convolution feature map을 flatten하여 사용.
- SL classifier: encoder feature 위에 linear classification head 사용.
- SSL projection head: `Linear(512*256, 512) -> ReLU -> Linear(512, 128)`.

Encoder는 convolution block에 넣기 전에 각 sample을 temporal standard deviation으로 정규화합니다.

### Contrastive SSL

SSL pretraining은 SimCLR-style NT-Xent loss를 사용합니다. 같은 원본 신호에서 만든 두 augmented view를 positive pair `(i, j)`로 두고, batch 안의 다른 view들을 negative로 사용합니다.

```text
Original I/Q signal
      |                         |
      v                         v
 RF augmentation view 1   RF augmentation view 2
      |                         |
      v                         v
 Shared 1D ResNet encoder Shared 1D ResNet encoder
      |                         |
      v                         v
 Projection head z_i      Projection head z_j
      \_________________________/
                 |
                 v
          NT-Xent contrastive loss
```

Loss는 다음과 같이 정리할 수 있습니다.

```text
L_i = -log exp(sim(z_i, z_j) / tau) / sum_{k != i} exp(sim(z_i, z_k) / tau)
```

여기서 `sim`은 cosine similarity이고, temperature `tau = 0.5`를 사용했습니다.

SSL augmentation은 RF 신호에 맞춘 perturbation으로 구성했습니다. 주로 additive noise, amplitude scaling, phase rotation, circular time shift를 사용합니다. 이후 fine-tuning 단계에서는 pretrained encoder 위에 supervised cross-entropy classifier를 붙여 학습합니다.

### Per-Head SSL

Per-head SSL은 하나의 공유 SSL encoder를 사용하되, classifier head를 SNR regime별로 분리합니다.

```text
Input I/Q signal
        |
        v
Shared SSL encoder
        |
        v
Encoder feature
        |
        +------------------> Severe head      (-10, -5 dB)
        |
        +------------------> Transition head  (0-5 dB)
        |
        +------------------> High head        (10, 15, 20 dB)
```

- severe head: `-10, -5` dB
- transition head: 학습 시 `0, 5` dB, dense 평가에서는 `0-5` dB 구간 담당
- high head: `10, 15, 20` dB

이 구조는 하나의 encoder representation 위에서 SNR 구간별 decision boundary를 따로 학습하는 것이 도움이 되는지 확인하기 위한 실험입니다.

## 폴더 구성

- `code/`: 신호 생성, SL/SSL 학습, dense SNR 평가, denoising SSL, per-head SSL 관련 Python 코드.
- `results/baseline_dense/`: SL과 contrastive SSL baseline의 dense SNR 평가 결과.
- `results/low_snr_exposure/`: low-SNR 데이터를 pretraining/fine-tuning에 추가한 ablation 결과.
- `results/per_head_ssl/`: SNR regime별 classifier head를 분리한 per-head SSL 결과.
- `results/denoising_ssl/`: MSE denoising SSL negative baseline 결과.
- `results/figures/`: 논문/포스터에 사용하기 위한 그림 파일.

## 주요 그림

논문에서 사용하는 핵심 비교 그림은 다음 파일입니다.

- `results/figures/figure_1_main_snr_comparison.png`
- `results/figures/figure_1_main_snr_comparison.pdf`
- `results/figures/figure_1_main_snr_comparison.svg`

이 그림은 label ratio `0.005`에서 다음 다섯 조건을 비교합니다.

- SL baseline
- contrastive SSL baseline
- pretraining과 fine-tuning 양쪽에 low-SNR 정보를 추가한 SSL
- SNR regime별 per-head SSL
- denoising SSL

## 주요 결과 파일

### Baseline Dense SNR

- `results/baseline_dense/sl_dense_eval_metrics.json`
- `results/baseline_dense/ssl_dense_eval_metrics.json`
- `results/baseline_dense/README.md`

SL baseline과 contrastive SSL baseline을 dense SNR test set에서 비교한 결과입니다.

### Low-SNR Exposure

- `results/low_snr_exposure/low_snr_exposure_metrics.json`
- `results/low_snr_exposure/README.md`

`-10`, `-5` dB 데이터를 pretraining 또는 fine-tuning에 추가했을 때 성능이 개선되는지 확인한 ablation입니다. 결과적으로 단순한 low-SNR exposure만으로는 severe low-SNR 성능이 유의미하게 개선되지 않았습니다.

### Per-Head SSL

- `results/per_head_ssl/per_head_multihead_metrics.json`
- `results/per_head_ssl/per_head_multihead_summary.json`
- `results/per_head_ssl/README.md`

공유 SSL encoder 위에 SNR regime별 classifier head를 분리한 실험입니다. severe low-SNR 자체를 해결하기보다는, 1-5 dB transition 구간과 high-SNR 구간에서 기본 SSL보다 더 나은 성능을 보였습니다.

### Denoising SSL

- `results/denoising_ssl/ssl_metrics.json`
- `results/denoising_ssl/README.md`

MSE 기반 waveform reconstruction을 사용하는 denoising SSL baseline입니다. 대부분 SNR에서 chance level 근처에 머물러, 단순 denoising objective가 변조 분류에 필요한 discriminative representation과 잘 맞지 않음을 보여주는 negative baseline으로 해석했습니다.

## 핵심 해석

1. Contrastive SSL은 모든 SNR에서 균일하게 성능을 올리는 방법은 아닙니다.
2. `-10`, `-5` dB의 severe low-SNR에서는 대부분의 방법이 chance level 근처에 머뭅니다.
3. 실제 성능 차이가 본격적으로 나타나는 구간은 대략 `1-5` dB의 transition SNR입니다.
4. Low-SNR 데이터를 단순히 더 보여주는 것만으로는 severe low-SNR 문제를 해결하지 못했습니다.
5. Per-head SSL은 SNR regime별 decision boundary를 다르게 학습할 수 있어 transition/high-SNR 구간에서 유용했습니다.
6. MSE denoising SSL은 본 과제에서는 효과적인 baseline이 아니었습니다.

## 제외한 파일

다음 파일은 GitHub 링크용 bundle에서 제외했습니다.

- raw/generated NumPy arrays (`*.npy`, `*.npz`)
- 모델 체크포인트 (`*.pth`, `*.pt`, `*.ckpt`)
- Colab 노트북
- 큰 zip archive
- CSV dump 및 train log
- Python cache 파일

이 구성은 논문 링크에서 코드, 핵심 JSON 결과, 그림, 실험 설명을 빠르게 확인하기 위한 목적입니다.

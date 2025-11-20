# Korean Spacing

경량 한국어 띄어쓰기 및 개행 교정 모델을 제공하는 npm 패키지입니다.

## 설치

```bash
npm install korean-spacing
```

## 사용법

### 기본 사용

```typescript
import KoreanSpacing from 'korean-spacing';

const spacing = new KoreanSpacing();
await spacing.load();

const result = await spacing.correct('안녕하세요반갑습니다');
console.log(result); // "안녕하세요 반갑습니다"

const multiline = await spacing.correct('첫번째문장입니다두번째줄도있습니다');
console.log(multiline);
// "첫번째 문장입니다\n두번째 줄도 있습니다"
```

### 배치 처리

```typescript
const texts = [
  '오늘날씨가좋네요',
  '한국어띄어쓰기는어렵습니다'
];

const results = await spacing.correctBatch(texts);
console.log(results);
// ["오늘 날씨가 좋네요", "한국어 띄어쓰기는 어렵습니다"]
```

## 모델 학습

### 1. 환경 설정

`uv`를 사용하여 Python 환경을 설정합니다:

```bash
# uv 설치 (아직 설치하지 않은 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Python 의존성 설치
npm run install:python
```

또는 직접:

```bash
uv sync
```

### 2. 모델 학습

허깅페이스 한글 데이터셋에서 공백/개행이 포함된 문장을 수집해 문자 단위 모델을 학습합니다:

```bash
npm run train
```

또는

```bash
python scripts/train.py
```

**데이터셋**: 기본적으로 아래 후보를 차례로 시도하며, 사용할 수 있는 첫 번째 데이터셋을 자동 선택합니다.

- `klue/ynat`
- `klue/nli`
- `klue/sts`
- `lcw99/wikipedia-korean-20221001`
- `HAERAE-HUB/KOREAN-WEBTEXT`

환경 변수 `DATASET_CANDIDATES`를 설정하면 후보를 직접 지정할 수 있습니다. 예)  
`DATASET_CANDIDATES="klue:ynat,lcw99/wikipedia-korean-20221001"`

`.env`를 사용하려면 `env.example`를 복사하여 필요한 값을 수정하면 됩니다.

```bash
cp env.example .env
# 필요 시 변수 수정
npm run train
```

병합 모드가 필요하면 `DATASET_MODE=concat`으로 설정해 여러 데이터셋을 한 번에 결합할 수 있습니다(기본값 fallback).

또한 `MAX_SAMPLES`, `TRAIN_BATCH_SIZE`, `TRAIN_EPOCHS` 등 환경 변수를 통해 데이터 크기와 학습 하이퍼파라미터를 조절할 수 있습니다.

**커리큘럼 학습**: 기본 설정은 짧은 문장(≤80자) → 중간 길이(≤160자) → 전체(≤512자) 순으로 단계별 학습을 진행합니다.  
환경 변수로 제어할 수 있습니다.

```
# 커리큘럼 비활성화
CURRICULUM_ENABLED=false

# 직접 단계 지정 (maxLength:epochs 목록)
CURRICULUM_SCHEDULE="80:2,200:1,512:2"
```

각 단계는 길이 조건을 만족하는 문장을 자동으로 필터링하며, 이전 단계 모델 가중치를 그대로 이어받아 학습합니다.
마지막에는 `all` 단계가 추가되어 1~3단계 데이터를 모두 섞어 한 번 더 미세 조정합니다.

**학습 안정화 옵션**

- `GRAD_CLIP_ENABLED` (기본 true), `GRAD_CLIP_MAX_NORM` (기본 1.0)으로 gradient clipping을 제어합니다.
- `AMP_ENABLED` (기본: CUDA 사용 시 자동 활성화)로 mixed precision을 켜/끌 수 있습니다.
- CUDA 환경에서는 `torch.backends.cudnn.benchmark = True`가 자동 설정되어 반복되는 입력 길이에서 조금 더 빠른 추론/학습이 가능합니다.
- 검증 로그에는 accuracy 외에 boundary-level F1 score도 함께 출력되어 공백/개행 예측 품질을 진단할 수 있습니다.
- `CONTINUE_TRAIN=true`로 설정하면 기존 체크포인트(`models/korean_spacing_char/char_spacing_model.pt`)를 불러와 이어서 학습합니다. 파일이 없으면 새로 학습을 시작합니다.

### 3. ONNX 변환

학습된 PyTorch 모델을 ONNX 형식으로 변환합니다:

```bash
npm run convert
```

또는

```bash
python scripts/convert_to_onnx.py
```

학습된 모델은 `./models/korean_spacing.onnx`에 저장되며, 원본 PyTorch 체크포인트와 구성 정보는 `./models/korean_spacing_char/`에 보관됩니다.

## 프로젝트 구조

```
korean-spacing/
├── src/
│   └── index.ts          # TypeScript 추론 코드
├── scripts/
│   ├── train.py          # 모델 학습 스크립트
│   └── convert_to_onnx.py # ONNX 변환 스크립트
├── models/               # 학습된 모델 저장 위치
├── package.json
└── README.md
```

## 모델 정보

- **입력 단위**: Unicode 문자 (패딩 포함 최대 512자)
- **라벨**: 0(없음), 1(공백), 2(개행)
- **구조**: 문자 임베딩 + BiLSTM + Linear
- **형식**: ONNX (경량화 추론) / PyTorch (학습)

## 개발

### 빌드

```bash
npm run build
```

### 예제 실행

```bash
npm run example
```

## 주의사항

- 입력 텍스트는 공백/개행이 제거된 문장이어야 합니다. 모델이 적절한 위치에 공백과 줄바꿈을 삽입합니다.
- 학습에 필요한 데이터는 허깅페이스 데이터셋에서 자동으로 수집하지만, 필요 시 `scripts/train.py`를 수정해 다른 데이터셋을 추가할 수 있습니다.
- `models/korean_spacing.onnx` 파일이 존재해야 TypeScript 추론 코드가 동작합니다.

## 라이선스

MIT


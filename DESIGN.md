# AI Service Benchmark 설계 문서

## 목적

"사내 AI 서비스가 느리다"는 말이 돌지만 근거가 없음.
동일 조건에서 응답 시간을 실측하여 **숫자 기반 근거**를 만드는 것이 목표.

## 비교 대상

| 엔드포인트 | 라이브러리 | 설명 |
|-----------|-----------|------|
| OpenAI 직접 | `openai` | OpenAI API 직접 호출 |
| AOAI (openai lib) | `openai` (`AzureOpenAI`) | Azure OpenAI를 openai 라이브러리로 호출 |
| AOAI (azure lib) | `azure-ai-inference` (`ChatCompletionsClient`) | Azure OpenAI를 Azure SDK로 호출 |

> 사내 AI 서비스는 웹 기반이라 API 비교 불가. 추후 별도 측정 예정.

## 모델

GPT-5 (OpenAI, AOAI 모두 동일 모델)

## 테스트 변수 (4개)

| 변수 | 값 |
|------|---|
| 엔드포인트/라이브러리 | OpenAI 직접 / AOAI(openai lib) / AOAI(azure lib) |
| API 방식 | Chat Completions / Responses API |
| 시스템 프롬프트 | 짧은(~50 tokens) / 긴 보안 한글(~2,000 tokens) |
| 코드 인터프리터 | OFF / ON |

**총 조합: 20가지** (azure-ai-inference는 Chat Completions만 지원 → 8+8+4)

## 질문 세트 (5개, 난이도 혼합)

모든 조합에 동일 질문 세트 사용.

| # | 난이도 | 질문 |
|---|-------|------|
| Q1 | 단답형 | 대한민국의 수도는? |
| Q2 | 설명형 | 인플레이션이 금리에 미치는 영향을 설명해줘 |
| Q3 | 분석형 | 금융회사에서 AI 도입 시 고려해야 할 리스크 3가지를 분석해줘 |
| Q4 | 코드 | Python으로 피보나치 수열 함수를 작성해줘 |
| Q5 | 복합 추론 | 연봉 5000만원 직장인의 소득세를 단계별로 계산해줘 |

## 반복 횟수

- 질문당 10회 반복
- 조합당 50회 (5문항 × 10회)
- **총 호출: 20 × 50 = 1,000회**

## 측정 지표

스트리밍 방식으로 측정.

| 지표 | 설명 |
|------|------|
| TTFT (Time To First Token) | 요청 → 첫 토큰 수신까지 |
| Total Latency | 요청 → 마지막 토큰 수신까지 |

`max_tokens` 제한 없음 (실제 사용 환경과 동일하게).

## 예상 비용

- GPT-5: Input $1.25/1M tokens, Output $10.00/1M tokens
- 예상 총 비용: **~9,400원 ($6.5)**

## 시스템 프롬프트

### 짧은 프롬프트

```
당신은 도움이 되는 AI 어시스턴트입니다.
```

### 긴 보안 프롬프트 (~2,000 tokens)

금융사 보안 컴플라이언스를 모사한 더미 프롬프트.
실제 사내 AI 서비스 시스템 프롬프트와 유사한 길이/구조.

내용:
- 역할 정의 및 행동 원칙
- 고객 개인정보 보호 지침
- 금융 상품 관련 응답 제한
- 투자 조언 면책 및 금지 사항
- 내부 시스템/인프라 정보 유출 방지
- 민감 데이터 처리 규정
- 응답 톤앤매너 가이드라인
- 법규 준수 (금융소비자보호법, 개인정보보호법 등)
- 에스컬레이션 절차
- 금지 행위 목록

## 결과 출력

### 형식
Markdown 파일 (`results/` 디렉토리) → GitHub에서 바로 확인 가능.

### 구성
변수별 그룹핑으로 **무엇이 느리게 하는 요소인지** 판별 가능하게:

1. **엔드포인트 비교** — OpenAI vs AOAI(openai) vs AOAI(azure)
2. **API 방식 비교** — Chat Completions vs Responses API
3. **시스템 프롬프트 영향** — 짧은 vs 긴 보안 프롬프트
4. **코드 인터프리터 영향** — OFF vs ON
5. **질문 난이도별 비교** — Q1~Q5

각 그룹에 포함되는 통계:
- 평균 (mean)
- 중앙값 (median)
- 최솟값 (min)
- 최댓값 (max)
- 표준편차 (std)

## 테스트 환경 (결과에 자동 기록)

| 항목 | 수집 방법 |
|------|----------|
| OS / 버전 | 자동 |
| Python 버전 | 자동 |
| `openai` 라이브러리 버전 | 자동 |
| `azure-ai-inference` 버전 | 자동 |
| 테스트 시각 | 자동 |
| Azure 리전 | 자동 (엔드포인트에서 추출) |

## 환경변수

```env
# OpenAI
OPENAI_API_KEY=

# Azure OpenAI
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT_NAME=
AZURE_OPENAI_API_VERSION=
```

## 파일 구조

```
ai-bench-kit/
├── DESIGN.md              # 이 문서
├── .env.example           # 환경변수 템플릿
├── .env                   # ← 여기에 키 넣으면 끝
├── prompts.md             # 시스템 프롬프트 원문 (사람이 읽기 편하게)
├── requirements.txt       # 의존성 (openai, python-dotenv)
├── benchmark.py           # 벤치마크 실행 (단일 파일)
└── results/               # 결과 마크다운 저장
    └── benchmark_YYYYMMDD_HHMMSS.md
```

## 실행 방법 (맥 기준, 딸깍)

```bash
# 1. 의존성 설치 (최초 1회)
pip install -r requirements.txt

# 2. .env 파일에 키 설정
cp .env.example .env
# .env 파일 열어서 키 입력

# 3. 실행
python benchmark.py
```

## 코드 원칙

- **단일 파일** (`benchmark.py`) — 불필요한 분리 없음
- **심플하게** — 과도한 추상화, 클래스 계층, 유틸리티 분리 금지
- **반복 최소화** — 라이브러리 차이 때문에 동일 로직 복붙하지 않음
- **누가 봐도 알기 쉽게** — 읽으면 바로 흐름이 보이는 구조

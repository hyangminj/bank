# bank

인구 밀도와 시설 밀도의 관계를 데이터 기반으로 분석하는 연구 레포지토리입니다.
본 프로젝트는 `dataset.tsv`를 사용해 전처리, KMeans 기반 군집 탐색, 그리고 분류 모델 평가를 수행합니다.

## 선행연구 (Prior Study)

이 저장소의 연구 방향은 아래 논문을 선행연구로 참고합니다.

- Um, J., Son, S.-W., Lee, S.-I., Jeong, H., & Kim, B. J. (2009). *Scaling laws between population and facility densities*. Proceedings of the National Academy of Sciences of the United States of America, 106(34), 14236-14240. https://doi.org/10.1073/pnas.0901898106

핵심 배경:

- 시설 밀도 `D`와 인구 밀도 `rho` 사이에는 양의 상관관계가 보고됨
- 단일 고정 지수만으로 설명되기보다 시설 유형에 따라 `D ~ rho^alpha`의 `alpha`가 달라질 수 있음
- 논문은 상업시설(수익 중심)과 공공시설(사회적 비용 중심)에서 서로 다른 스케일링 양상을 제시함

## 프로젝트 목표

- 범주형/연속형 특성을 함께 전처리해 시설 관련 목표 변수를 분류
- 인구 및 사회경제 특성을 바탕으로 군집 구조를 탐색
- 모델별 교차검증 성능을 비교해 데이터 구조를 해석

## 데이터

- 파일: `dataset.tsv`
- 형식: 탭(`\t`) 구분, 헤더 없는 16개 컬럼
- 인코딩: UTF-8 가정 (한국어 범주값 포함 가능)

## 실행 방법

### 1) 환경 준비

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 기본 실행

```bash
python3 bank.py
```

### 3) 옵션 실행

```bash
python3 bank.py --data dataset.tsv --k-list 3,4,5,6 --cv 5 --seed 42 --output results.json
```

주요 옵션:

- `--data`: 입력 TSV 경로
- `--k-list`: KMeans에서 탐색할 k 목록(쉼표 구분)
- `--cv`: 교차검증 fold 수
- `--seed`: 난수 시드
- `--output`: 결과 JSON 저장 경로

## 현재 파이프라인

1. 데이터 로드 및 컬럼 개수 검증
2. 범주형(one-hot) + 연속형(min-max) 전처리
3. 초기 특성(앞 8개 컬럼) 기반 KMeans 군집 탐색
4. 전체 특성(목표 제외 15개 컬럼) 기반 분류 모델 평가
   - Decision Tree
   - Random Forest
   - Extra Trees

## 해석 시 유의사항

- 성능 수치는 데이터 분할/시드/전처리 정의에 따라 달라질 수 있음
- 본 코드는 연구용 재현 실험을 위한 기준 구현이며, 운영 환경 최적화가 목적은 아님

## License

이 프로젝트는 학술 연구(academic research) 목적으로만 사용 가능합니다.
상업적 사용 및 비학술 목적 사용은 허용되지 않습니다.
자세한 내용은 `LICENSE` 파일을 참고하세요.

# 🧠 Layer 2: Spatial Logic Gate Engine

본 프로젝트의 핵심은 객체 탐지 결과를 후처리하여 실제 위반 여부를 판별하는 기하학 기반 공간 필터링 알고리즘입니다.
---

## 🔹 Why Post-Processing is Necessary?

YOLO 단독 출력만으로는 다음과 같은 문제가 발생합니다.

* 사람과 킥보드가 가까이 있어도 실제 탑승 관계인지 판단 불가
* 상반신/전신 중복 검출 발생
* 원근 효과로 인해 겹쳐 보이는 객체 오판 가능
* 헬멧 객체의 실제 소유자 연결 어려움

이를 해결하기 위해 Layer 2 추론 엔진을 설계했습니다.

---

## 🛠️ 비정상 주행(Driving-Abnormal) 판별 로직의 진화 과정

### 📍 1차 안: 단순 객체 카운트 (Baseline)
* **규칙:** $\text{human Count} > \text{kickboard Count} \implies \text{Abnormal (다인승)}$
* **한계:** 킥보드와 무관한 배경의 일반 행인까지 카운트되어 정상 주행을 위반으로 오진(False Positive)하는 치명적 오류 발생.

### 📍 2차 안: 공간 좌표(Y축 원근법) 기반 행인 필터링
* **개념:** 킥보드 하단 좌표와 사람 하단 좌표 간의 수직 거리(Distance) 비교를 통해 1차 필터링.
* **한계:** 이미지 스케일 변화에 따른 고정 임계값 설정의 어려움 및 다인승 탑승 시 발 위치 편차로 인한 오류 존재.

### 📍 3차 최종안: 공간 교집합(IoU) 및 발끝 종속성(Foot-Point Dependency) 복합 연산
1. **바닥 타이트 매칭 (Tight Boundary Match):** `human` 박스의 하단 좌표($Y_{\text{max}}$)가 킥보드의 동적 마진(Margin) 내에 위치하는지 검증하여 원거리 객체 및 배경 행인 완벽 차단.
2. **IoU-Based Relationship:** 겹쳐 타는 다인승의 특성을 반영하여 사람과 킥보드, 그리고 탑승자 간의 교집합 비율 수학적 연산.
3. **Helmet Association:** 탑승자로 판별된 객체의 상단 영역에 `helmet` 객체가 매칭되는지 확인.

---

## ✅ 최종 의사결정 파이프라인 (Decision Pipeline)
특정 킥보드에 매칭된 탑승자 그룹이 확정되면 다음 규칙을 적용합니다.

1. **위험 주행 (다인승):** 킥보드에 매칭된 `human` $\ge 2$
2. **위험 주행 (헬멧 미착용):** 탑승자는 있으나 상단 영역에 `helmet` 미탐지 (`bare_head` 탐지)
3. **정상 주행:** 위 조건에 해당하지 않는 안전한 1인 탑승 상태

---
# ⚙️ Core Spatial Reasoning Algorithm

## 1️⃣ IoU-Based Relationship Filtering

사람(`human`)과 킥보드(`kickboard`) 간의 IoU(Intersection over Union)를 계산하여
실제 탑승 관계인지 검증합니다.

```python
IoU = Intersection Area / Union Area
```

일정 임계값 이상의 교집합이 발생할 경우에만
탑승자로 간주합니다.

---

## 2️⃣ Foot-Point Dependency Verification

단순 IoU만 사용할 경우
원근 왜곡(Perspective Distortion)에 의해 오탐지가 발생할 수 있습니다.

이를 방지하기 위해 사람 Bounding Box의 하단 중심 좌표(Foot Point)가
킥보드 영역과 논리적으로 연결되는지를 추가 검증합니다.

```text
Human Foot Point
        │
        ▼
Kickboard Region
```

이 방식은:

* 배경 보행자 제거
* 원거리 객체 오탐 감소
* 겹쳐 보이는 객체 분리

에 매우 효과적입니다.

---

## 3️⃣ Helmet Association Logic

검출된 `helmet` / `bare_head` 객체를
탑승자(`human`)와 매칭하여 안전모 착용 여부를 판단합니다.

### 판별 기준

* `helmet` 존재 → 정상 탑승
* `bare_head` 존재 → 안전모 미착용 위반
* 다인 탑승 탐지 시 → 중복 위반 처리 가능

---
## 📊 Results & Discussion
별도로 구축한 알고리즘 테스트 데이터셋(총 ??장)을 대상으로 최종 판별을 진행한 결과입니다.

| 실제 상태 (Ground Truth) | 모델 판별: NORMAL | 모델 판별: ABNORMAL | 판별 성공률 |
| :--- | :---: | :---: | :---: |
| **정상 주행 (NORMAL / ??장)** | **?? 장** <br>*(True Normal)* | **0 장** <br>*(False Positive)* | **100%** |
| **위반 주행 (ABNORMAL / ??장)** | **?? 장** <br>*(False Negative)* | **?? 장** <br>*(True Abnormal)* | **?? %** |

### 💡 한계점 및 로직 타당성 고찰
본 프로젝트는 보수적인 기준을 적용하여, **정상 이용자를 위반자로 오인하는 치명적인 오류(False Positive)를 원천 차단(0%)**하는 데 성공했습니다. 

1. **미탐지 원인 (False Negative):** 두 탑승자가 완벽히 밀착되거나 다리가 벌어진 기형적 자세 등, 절대적인 위반 이미지 학습량 부족 및 2D 원근법의 태생적 한계로 인해 일부 위반 차량을 정상으로 오인하는 경우가 존재합니다.
2. **실서비스 타당성:** 공유 모빌리티 자동 단속 시스템에서 선량한 이용자에게 패널티가 부과되는 시스템 오류(`False Positive`)는 서비스 신뢰도에 치명적입니다. 따라서 특이 케이스 위반자를 일부 놓치더라도(`False Negative`), **정상 사용자를 철저하게 보호하는 현재의 보수적 알고리즘 설계는 본 프로젝트의 운용 목적에 완벽하게 부합**합니다.

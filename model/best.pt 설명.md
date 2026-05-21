# PM Security: Personal Mobility Safety Violation Detection System

> **딥러닝 기반 전동 킥보드 안전 수칙 위반 자동 감지 시스템**
> *Real-time Personal Mobility (PM) Safety Violation Detection with Spatial Reasoning Pipeline*

---

## 📌 Overview

본 프로젝트는 전동 킥보드(Personal Mobility, PM)의 안전 수칙 위반 행위를 자동으로 감지하기 위한 **AI 기반 실시간 영상 분석 시스템**입니다.

단순 객체 탐지(Object Detection)에 그치지 않고,
검출된 객체 간의 **공간적 관계(Spatial Relationship)** 를 추가적으로 분석하는 **2-Stage 추론 구조**를 설계하여 다음과 같은 실제 도로 환경 문제를 해결했습니다.

* 다인 탑승 감지
* 헬멧 미착용 감지
* 원근 왜곡(Perspective Distortion) 대응
* 상반신/전신 중복 검출 제거
* 오탐(False Positive) 최소화

특히 본 프로젝트의 핵심은
YOLO 기반 객체 탐지 결과를 그대로 사용하는 것이 아니라,
후처리 단계에서 **기하학적 공간 연산 알고리즘(Geometric Spatial Filtering)** 을 적용하여 실제 위반 상황만을 정교하게 판별하는 데 있습니다.

---

# 🏗️ System Architecture

시스템은 다음과 같은 **Two-Layer 구조**로 설계되었습니다.

```text
Input Image
      │
      ▼
┌────────────────────┐
│ Layer 1            │
│ YOLO26n Detection  │
│ (Object Detection) │
└────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│ Layer 2                     │
│ Spatial Logic Gate Engine   │
│ - IoU Filtering             │
│ - Foot-Point Dependency     │
│ - Geometric Validation      │
└─────────────────────────────┘
      │
      ▼
NORMAL / VIOLATION
```

---

# 🧩 Layer 1 — Object Detection Engine

실시간 엣지 환경(CCTV, 스마트 교통 카메라 등)에서의 동작을 고려하여
경량 고속 모델인 **YOLO26n** 기반으로 학습을 수행했습니다.

## 🔹 Detection Classes

| Class ID | Label       | Description |
| -------- | ----------- | ----------- |
| 0        | `bare_head` | 안전모 미착용 머리  |
| 1        | `helmet`    | 안전모 착용 상태   |
| 2        | `human`     | 탑승자 신체      |
| 3        | `kickboard` | 전동 킥보드      |

---

# 📂 Dataset & Annotation

## 🔹 Annotation Tool

* Roboflow 기반 고정밀 Bounding Box Annotation 수행

## 🔹 Dataset Split

과적합(Overfitting)을 방지하고 객관적인 성능 검증을 위해
데이터셋을 완전히 분리하여 학습했습니다.

| Dataset    | Ratio |
| ---------- | ----- |
| Train      | 80%   |
| Validation | 10%   |
| Test       | 10%   |

---

# 🚀 Model Training

## 🔹 Training Configuration

| Parameter            | Value        |
| -------------------- | ------------ |
| Base Model           | `yolo26n.pt` |
| Image Size           | `640 × 640`  |
| Epochs               | `50`         |
| Device               | `CUDA GPU`   |
| Confidence Threshold | `0.3`        |
| IoU Threshold        | `0.45`       |

## 🔹 Optimization Strategy

### ✔ Confidence Threshold Tuning

낮은 신뢰도의 노이즈 박스를 제거하여 오탐(False Positive)을 감소시킴

### ✔ IoU-Based Duplicate Suppression

상반신/전신 중복 박스 문제를 방지하여 실제 객체 수를 안정적으로 유지

### ✔ Edge-Oriented Lightweight Inference

실시간 추론 속도를 확보하기 위해 경량 아키텍처 사용

---

# 📊 Final Performance (Test Dataset)

학습에 전혀 사용되지 않은 순수 Test Dataset 기준 성능입니다.

| Metric    |      Score |
| --------- | ---------: |
| mAP50     | **95.92%** |
| Precision | **88.56%** |
| Recall    | **89.20%** |

---

# 🧠 Layer 2 — Spatial Logic Gate

본 프로젝트의 핵심 기여(Core Contribution)는
객체 탐지 결과를 후처리하여 실제 위반 여부를 판별하는
**기하학 기반 공간 필터링 알고리즘**입니다.

---

## 🔹 Why Post-Processing is Necessary?

YOLO 단독 출력만으로는 다음과 같은 문제가 발생합니다.

* 사람과 킥보드가 가까이 있어도 실제 탑승 관계인지 판단 불가
* 상반신/전신 중복 검출 발생
* 원근 효과로 인해 겹쳐 보이는 객체 오판 가능
* 헬멧 객체의 실제 소유자 연결 어려움

이를 해결하기 위해 Layer 2 추론 엔진을 설계했습니다.

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

# ✅ Final Decision Pipeline

```text
[Detection Results]
        │
        ▼
Human ↔ Kickboard Matching
        │
        ▼
Spatial Validation
(IoU + Foot Dependency)
        │
        ▼
Helmet Association
        │
        ▼
NORMAL / VIOLATION
```

---

# 🎯 Key Contributions

## ✔ 단순 객체 탐지를 넘어선 공간 추론 기반 설계

기존 YOLO 출력 결과를 그대로 사용하지 않고
공간적 관계를 수학적으로 분석하여 실제 상황을 해석

## ✔ 원근 왜곡 방어 알고리즘 적용

Foot-Point 기반 검증으로 CCTV 환경 오탐 최소화

## ✔ Edge Device 실시간 운용 가능

경량 모델 기반으로 실시간 추론 환경 고려

## ✔ 실제 서비스형 구조 설계

단순 연구용 모델이 아닌:

* CCTV
* 스마트 시티
* 교통 안전 시스템
* 공유 모빌리티 관제

등 실제 서비스 적용을 고려한 파이프라인 구성

---

# 🛠️ Tech Stack

| Category         | Stack    |
| ---------------- | -------- |
| Deep Learning    | YOLO26   |
| Framework        | PyTorch  |
| Annotation       | Roboflow |
| Language         | Python   |
| GPU Acceleration | CUDA     |
| Computer Vision  | OpenCV   |

---

# 📈 Future Improvements

* Multi-Object Tracking (MOT) 적용
* 영상 기반 Temporal Consistency 검증
* 야간/우천 환경 데이터 증강
* ONNX/TensorRT 최적화
* 실시간 스트리밍 서버 연동
* Edge AI Deployment

---

# 👨‍💻 Author

**PM Security Project**
AI-Based Personal Mobility Safety Monitoring System

---

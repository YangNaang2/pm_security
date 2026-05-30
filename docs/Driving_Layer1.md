# 🚀 Layer 1: Object Detection Model Training

본 문서는 실시간 엣지 환경(CCTV, 스마트 교통 카메라 등)을 고려하여 구성한 **YOLO26n 기반 객체 탐지 엔진(Layer 1)**의 학습 과정과 성능 지표를 다룹니다.

## 📂 Dataset & Annotation
* **Annotation Tool:** Roboflow 기반 고정밀 Bounding Box Annotation 수행
* **Dataset Split:** 과적합 방지 및 객관적 성능 검증을 위해 **8:1:1 (Train:Validation:Test)** 비율로 데이터를 완전 분리 격리

### 🔹 Detection Classes
| Class ID | Label       | Description |
| -------- | ----------- | ----------- |
| 0        | `bare_head` | 안전모 미착용 머리  |
| 1        | `helmet`    | 안전모 착용 상태   |
| 2        | `human`     | 탑승자 신체      |
| 3        | `kickboard` | 전동 킥보드      |

## ⚙️ Training Configuration
| Parameter            | Value        |
| -------------------- | ------------ |
| Base Model           | `yolo26n.pt` |
| Image Size           | `640 × 640`  |
| Epochs               | `50`         |
| Device               | `CUDA GPU`   |

## 📊 Final Performance (Layer 1)
학습에 전혀 사용되지 않은 순수 **Test Dataset** 기준 객체 탐지 성능입니다.

| Metric    |      Score |
| --------- | ---------: |
| mAP50     | **95.92%** |
| Precision | **88.56%** |
| Recall    | **89.20%** |

### 💡 Layer 1 성능 분석 (Performance Analysis)

엣지 디바이스용 경량화 모델(YOLO26n)을 사용했음에도 불구하고, **실서비스 적용 가능 수준의 검출력**이 나왔습니다.

* **압도적인 객체 인식률 (mAP50 95.92%):** 시스템의 '눈' 역할을 하는 기초 탐지 모듈이 도로 환경의 다양한 변수 속에서도 킥보드와 사람, 헬멧을 거의 놓치지 않고 정확하게 식별해 냅니다.
* **정밀도(Precision)와 재현율(Recall)의 밸런스:** 88~89% 대의 균형 잡힌 지표는 모델의 False Positive와, False Negative 사이에서 밸런스를 유지하고 있음을 보여줍니다.
* **Layer 2 알고리즘을 준비 완료:** 단순 객체 탐지를 수행하는 Layer 1의 성능이 안정적으로 확보되었습니다. YOLO의 Bounding Box를 신뢰하고 알고리즘을 설계할 수 있는 환경이 구축되었습니다.

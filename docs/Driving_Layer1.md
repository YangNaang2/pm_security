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

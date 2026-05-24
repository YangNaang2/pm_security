# 🛴 pm_security

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Ultralytics-YOLO26-FF6F00?style=flat-square&logo=ultralytics&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=flat-square&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
</p>

> **2026년 1학기 한양대학교 ERICA 텐서프로그래밍 기말 프로젝트** 

> 객체인식을 활용한 개인형 이동장치(PM)의 불법주차 및 불법주행 자동 단속 시스템

## 📌 Project Overview
최근 도심 내 공유 킥보드와 자전거(PM)의 무분별한 무단 방치와 불법 주행으로 인한 보행 안전 문제가 심각해지고 있습니다. 특히 한양대 에리카캠퍼스가 위치한 안산시는 경기도 내 개인형 이동장치(PM) 사고 발생 건수 1위를 기록했으며, 2026년 5월부터는 특정 구역에서는 즉시 견인 방침을 발표하기도 했습니다. 본 프로젝트는 **객체 인식**을 활용하여 <u>안전을 위협하는 고위험 불법 주차 및 주행 행위</u>를 실시간으로 탐지하고 관리하는 시스템의 프로토타입을 제안합니다.

## 🎯 주요 탐지 대상
* **불법 주차:** 횡단보도 3m 이내, 점자블록 위, 도로 위, 지하철역 주변 등 보행 및 차량 안전을 직접 위협하는 구역
* **불법 주행:** 헬멧 미착용, 2인 이상 탑승 등 육안으로 식별 가능한 위험 주행


## 🏗️ System Architecture (Brief) / Driving Part
본 시스템은 단순 객체 탐지의 한계를 극복하기 위해 **Two-Layer 구조**로 설계되었습니다.
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
1. **Layer 1 (Object Detection):** YOLO26n 기반 4가지 클래스(`human`, `kickboard`, `helmet`, `bare_head`) 탐지
2. **Layer 2 (Spatial Logic Gate):** 탐지된 객체 간의 공간적 교집합(IoU) 및 원근법 종속성을 수학적으로 연산하여 최종 위반 판별


## 📚 상세 기술 문서 (Documentation)
프로젝트의 세부 로직 및 성능 지표는 아래 링크를 확인해 주세요.
* [**[Layer 1] 모델 학습 및 성능 지표 상세 보기**]([docs/model_training.md](https://github.com/tensor-programming-2026/pm_security/blob/main/docs/Driving_Layer1.md))
* [**[Layer 2] 기하학 기반 공간 필터링 알고리즘 및 한계점 상세 보기**]([docs/algorithm_logic.md](https://github.com/tensor-programming-2026/pm_security/blob/main/docs/Driving_Layer2.md))

  
## 🚀 Quick Start
```
conda create -n pm_security python=3.11 -y
conda activate pm_security
```
아래 명령어 실행 전 torch, torchvision은 사용환경에 맞게 먼저 설치해주세요.

```
pip install -e .
```

* [주행 모델 다인승 판별 알고리즘 상세 보기](docs/algorithm_logic.md)
## 💻 실행 방법

본 프로젝트의 모든 핵심 파이프라인(데이터 전처리, 추론, 정량 평가)은 Google Colab 환경에서 원클릭으로 재현할 수 있도록 주피터 노트북(`.ipynb`) 파일에 모두 통합되어 있습니다.

### 📌 Step 1. 사전 데이터 준비
1. 학습이 완료된 모델 가중치 파일(`best.pt`)과 이미지 데이터셋 압축 파일(`dataset_3.yolo26.zip`)을 다운로드합니다.
2. 위 두 파일을 본인의 **구글 드라이브 내 특정 폴더**(예: `MyDrive/pm_security/`)에 미리 업로드해 둡니다.

### 📌 Step 2. 노트북 파일 환경 세팅
1. 본 Repository의 model에 첨부된 `1st_test.ipynb`을 컴퓨터에 다운로드합니다.
2. Google Colab에 접속한 뒤 다운로드한 `.ipynb` 파일을 엽니다.

### 📌 Step 3. 파이프라인 실행
업로드된 코랩 환경에서 위에서부터 순서대로 코드 셀을 실행(Shift + Enter)하거나, 상단의 [런타임] ➡️ [모두 실행]을 클릭합니다. 노트북 내부에는 다음의 전체 과정이 자동화되어 있습니다.

1. **환경 세팅:** 구글 드라이브 마운트 및 데이터셋 로컬 초고속 복사 (최초 1회 구글 로그인 권한 승인 필요)
2. **데이터 전처리:** `test` 데이터셋 8:1:1 자동 분할 및 메타데이터(`data.yaml`) 동적 업데이트
3. **알고리즘 추론:** YOLO26n 기반 1차 객체 탐지(NMS 적용) 및 2차 기하학적 공간 필터링 알고리즘 가동
4. **결과 시각화:** 실시간 위반 차량 적발 화면 및 최종 성능 지표(mAP, 혼동 행렬) 출력

> **💡 TroubleShooting:** 만약 파일을 찾을 수 없다는 경로 에러가 발생할 경우, 노트북 최상단 셀의 구글 드라이브 복사 경로(`!cp /content/drive/MyDrive/pm_security/...`)가 본인이 실제로 드라이브에 파일을 업로드한 경로와 일치하는지 확인해 주세요.

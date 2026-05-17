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

## 🎯 주요 탐지 대상 (데이터셋 기준)
* **불법 주차:** 횡단보도 3m 이내, 점자블록 위, 도로 위, 지하철역 주변 등 보행 및 차량 안전을 직접 위협하는 구역
* **불법 주행:** 헬멧 미착용, 2인 이상 탑승 등 육안으로 식별 가능한 위험 주행

## 🚀 Quick Start

```
conda create -n pm_security python=3.11 -y
conda activate pm_security
pip install .
```
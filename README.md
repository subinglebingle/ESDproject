# ESDproject

## dataset

* data_resize: 원본 데이터를 640x640의 해상도로 변경
  
* data_label: labelImage 프로그램을 사용하여 라벨링됨
  

## 성능 평가를 위해서 '비교' 시스템 구축

  기본 YOLO
  
  기본 YOLO+ 추가 데이터 학습
  
  기본 YOLO+ ROI설정
  
  기본 YOLO+ 추가 데이터 학습+ ROI설정 (*최종 목표)
  
  비교모델
  
  비교모델+ 추가 데이터 학습
  

## 계획
8주차: 데이터 수집 (70%)

9주차: 데이터 수집, ROI 설정한 YOLO모델 구현 (95%)

>	Fps, 최적화 등 코드 보완 필요

10주차: 데이터 수집, 라즈베리파이에서의 buzzer, picamera 코드 작성 (100%)

11주차: FPS 향상을 위한 코드 보완, 실제 라즈베리파이로 실험(80%)

>	예상보다 라즈베리파이에서의 fps가 너무 낮음. 추가적인 보완 필요.

12주차: (기말고사),  기존연구에서 비교모델 찾기

13주차: 코너 케이스에 대한 ROI 코드 설정

14주차: 결과정리, 비교, 분석


## 중간결과

노트북 & 내장웹캠
![Image](https://github.com/user-attachments/assets/fc4f456f-3b0d-4c6a-981f-c8e199b8afdd)

라즈베리파이 & picamera
![Image](https://github.com/user-attachments/assets/e3dd7413-1ae2-46ce-8cd6-c850b1fba399)

# ESDproject

This repository contains the code for 'Embedded System Design' course project.

Our model, based on YOLOv5, detects beverages and triggers a buzzer alarm upon detection.
We use MediaPipe to recognize hands and set a ROI to improve accuracy.

This system is expected to be applicable in various environments such as hospitals, public transportation, and libraries.

## Index

[🗺️Roadmap](##-Roadmap)

[📝Change Log](##-Change-log)

[📂Dataset](##-Dataset)

[🧠Models for Comparative Evaluation](##-Models-for-Comparative-Evaluation)

[🔬Experiments](##-Experiments)



## 🗺️Roadmap
8주차: 데이터 수집

9주차: 데이터 수집, ROI 설정한 YOLO모델 구현

>	Fps, 최적화 등 코드 보완 필요

10주차: 데이터 수집, 라즈베리파이에서의 buzzer, picamera 코드 작성

11주차: FPS 향상을 위한 코드 보완, 실제 라즈베리파이로 실험

>	예상보다 라즈베리파이에서의 fps가 너무 낮음. 추가적인 보완 필요.

12주차: (기말고사),  기존연구에서 비교모델 찾기

13주차: 코너 케이스에 대한 ROI 코드 설정

> 코너케이스와 낮은 fps를 보완하기 위해 MediaPipe 활용

14주차: 결과정리, 비교, 분석


## 📝Change log

[25/05/05] Uploaded custom dataset: 233 images with labels.

[25/05/14] Uploaded inference code for Raspberry Pi and PiCamera.

>Includes ROI setting, buzzer activation upon detecting 'drinks' and FPS display functionality.

[25/05/15] Uploaded training code, including the trained weights best.pt.

[25/05/26] Add 46 images with labels (total)

## 📂Dataset

* custom data: 279개의 직접 수집한 이미지와 라벨

    * data_resize: 원본 데이터를 640x640의 해상도로 변경
  
    * data_label: labelImage 프로그램을 사용하여 라벨링됨

* 최종 모델은 coco dataset 중 'bottle'과 'cup'이 포함된 1352개의 데이터를 커스텀 데이터에 추가하여 **1631개의 데이터**로 학습
  

## 🧠Model for Comparative Evaluation
  
  기본 YOLO+ 커스텀 데이터

  기본 YOLO+ 커스텀 & coco 데이터 학습
  
  기본 YOLO+ 커스텀 & coco 데이터 학습+ ROI설정 (*최종 목표)
  

## 🔬Experiments

<details>
  <summary>중간결과</summary>

노트북 & 내장웹캠

![Image](https://github.com/user-attachments/assets/fc4f456f-3b0d-4c6a-981f-c8e199b8afdd)

라즈베리파이 & picamera

![Image](https://github.com/user-attachments/assets/e3dd7413-1ae2-46ce-8cd6-c850b1fba399)

</details> 

---

### 최종결과

<details>
<summary>📊 Custom Dataset (Orange) vs Custom + COCO Dataset (Green)</summary>

* train & val set  
![Image](https://github.com/user-attachments/assets/8f81fe99-2606-4739-9337-72dd93f46ae8)

* recall, precision, mAP  
![Image](https://github.com/user-attachments/assets/1c914f0a-e66d-4773-b280-aaedfe67326b)

</details>


<details>
<summary>🎥 Model Demo Video</summary>

* **Final Model**  
  ![Image](https://github.com/user-attachments/assets/a057fc08-da42-4771-b00f-6004a60cbd4b)  
  ![Image](https://github.com/user-attachments/assets/78a53ba7-87fb-4dd6-a1ad-d0710e6cb37a)

* Model Behavior Without a Drink    
  ![Image](https://github.com/user-attachments/assets/f3754cc3-ccf6-4c57-aecd-0c1795e5300e)

* Model Performance on Corner Case   
  ![Image](https://github.com/user-attachments/assets/bb3400a9-2714-4aff-8fcf-27467c2150ac)

* Comparison Model: Without ROI    
  ![Image](https://github.com/user-attachments/assets/12580495-fa02-4e9f-9c3a-5722016f93e2)  
  ![Image](https://github.com/user-attachments/assets/f8da88cf-784d-444e-a959-7a1d92ed3464)

</details>



<details>
<summary>📷 Recommended Camera Installation Environment</summary>

![Image](https://github.com/user-attachments/assets/9cd823a8-c96f-45ff-9fe8-a5d4332f915f)  
![Image](https://github.com/user-attachments/assets/fa1b72c7-9eb1-470d-92fd-94f2550e0936)  
![Image](https://github.com/user-attachments/assets/31e0666b-5520-4d71-9fdb-49232b3628a7)

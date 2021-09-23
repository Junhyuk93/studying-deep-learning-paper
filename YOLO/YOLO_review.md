# YOLO Review

---

# 1. Abstract 

- We present YOLO, a new approach to object detection.
YOLO 연구진은 object detection에 새로운 접근방식을 적용.

- Prior work on object detection repurposes classifiers to perform detection.
이전의 multi-task 문제를 하나의 회귀 문제로 재정의함.

- A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation.
이미지 전체에 대해서 하나의 신경망이 한번의 계산만으로 bounding box와 class를 예측함.

- Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.
하나의 신경망으로 구성되어 있으므로 end-to-end 형식임.

- Our unified architecture is extremely fast. Our base YOLO model processes images in real-time at 45 frames per second.
우린 정말 빠르다.


## 2. Introduction
기존의 검출 모델은 classifier를 재정의하여 detector로 사용함. 분류란 하나의 이미지를 보고 그것이 개인지 고양이 인지 판단하는 것을 뜻함. 하지만 객체 검출(object detection)은 하나의 이미지 내에서 개는 어디에 위치해 있고, 고양이는 어디에 위치해 있는지 판단하는 것임. 따라서 객체 검출은 분류뿐만 아니라 위치 정보도 판단해야 함. 기존의 객체 검출 모델로는 대표적으로 [DPM](https://www.cs.cmu.edu/~deva/papers/dpm_acm.pdf)과 [R-CNN](https://arxiv.org/pdf/1311.2524.pdf)이 있음.

Deformable parts models(DPM)은 이미지 전체를 거쳐 sliding window 방식으로 객체 검출을 하는 모델.
R-CNN은 이미지 안에서 bounding box를 생성하기 위해 region proposal이라는 방법을 사용. 그렇게 제안된 bounding box에 classifier를 적용하여 분류함. 분류한 뒤, bounding box를 조정하고, 중복된 검출을 제거하고, 객체에 따라 box의 점수를 재산정하기 위해 post-processing을 함. 이런 복잡성 때문에 R-CNN은 느리고 절차를 독립적으로 훈련시커야 하므로 최적화가 힘듬.

그리하여 객체 검출을 하나의 회귀 문제로 보고 절차를 개선함. 임지ㅣ의 픽셀로부터 bounding box의 위치, 클래스 확률을 구하기까지의 일련의 절차를 하나의 회귀 문제로 정의.
> **이러한 시스템을 통해 YOLO는 이미지 내에 어떤 물체가 있고 그 물체가 어디에 있는지를 하나의 파이프 라인으로 빠르게 구해줌.**

![](https://i.imgur.com/RsCml1f.png)

하나의 convolutional network가 여러 bounding box와 그 bounding box의 클래스 확률을 동시에 계산해줌. YOLO는 이미지 전체를 학습하여 곧바로 검출 성능(detection performance)를 최적화함.

1. **엄청 빠름**. 기존의 복잡한 객체 검출 프로세르를 하나의 회귀 문제로 바꾸었기 때문. [참고영상](https://www.youtube.com/watch?v=MPU2HistivI)
![](https://i.imgur.com/K3Nbt3h.gif)
2. **예측을 할 때 이미지 전체를 봄**. sliding window나 region proposal 방식과 달리, YOLO는 훈련과 테스트 단계에서 이미지 전체를 봄. 그리하여 클래스의 모양에 대한 정보뿐만 아니라 주변 정보까지 학습하여 처리함.
3. **YOLO는 물체의 일반적인 부분을 학습함**. 일반적인 부분을 학습하기 때문에 자연 이미지를 학습하여 그림 이미지로 테스트할 때, DPM이나 R-CNN보다 월등히 뛰어남. 다른 모델에 비해 새로운 이미지에 대해 더 강건(robust)함. 즉 검출 정확도가 더 높음.

하지만 YOLO는 SOTA 객체 검출 모델에 비해 정확도가 다소 떨어짐. 빠르게 객체를 검출할 수 있다는 장점이 있지만 작은 물체에 대한 검출 정확도가 떨어짐.


---
    2. Introduction 요약: 
     - YOLO는 단일 신경망 구조이기 때문에 구성이 단순하며, 빠르다.
     - YOLO는 주변 정보까지 학습하며 이미지 전체를 처리하기 때문에 background error가 작다.
     - YOLO는 훈련 단계에서 보지 못한 새로운 이미지에 대해서도 검출 정확도가 높다.
     - 단, YOLO는 SOTA 객체 검출 모델에 비해 정확도(mAP)가 다소 떨어진다.
--- 

## 3. Unified Detection

YOLO는 객체 검출의 개별 요소를 단일 신경망(single neural network)으로 통합한 모델임.
YOLO는 입력 이미지(input images)를 S x S 그리드(S x S grid)로 나눔. 만약 어떤 객체의 중심이 특정 그리드 셀(grid cell)안에 위치한다면, 그 그리드 셀에 해당 객체를 검출해야 함.

각각의 그리드 셀(gird cell)은 B개의 bounding box와 그 bounding box에 대한 confidence score를 예측함.
confidence score는 bounding box가 객체를 포함한다는 것을 얼마나 믿을만한지, 그리고 예측한 bounding box가 얼마나 정확한지를 나타냄.

![](https://i.imgur.com/ZyqNthx.png)


이 score 는 **bounding box에 특정 클래스(class) 객체가 나타날 확률**(=Pr(Class_i))과 **예측된 bounding box가 그 클래스(class) 객체에 얼마나 잘 들어맞는지(fits the object)**(=OUT_pred^truth)를 나타냄.

![](https://i.imgur.com/mGZgtBu.png)


---
    3. Unified Detection 요약:
      - YOLO는 객체 검출의 개별 요소를 단일 신경망(single neural network)으로 통합한 모델이다.
      - 입력 이미지(input images)를 S x S 그리드(S x S grid)로 나눈다.
      - 각각의 그리드 셀(grid cell)은 B개의 bounding box와 그 bounding box에 대한
        confidence score를 예측한다.
      - class-specific confidence score는 bounding box에 특정 클래스(class) 객체가
        나타날 확률과 예측된 bounding box가 그 클래스(class) 객체에 얼마나 잘 들어맞는지를 나타낸다.
      - 최종 예측 텐서의 dimension은 (7 x 7 x 30)이다.
---

### 3.1 Network Design

![](https://i.imgur.com/2EpSoJw.png)

### 3.2 Training

---
    3.2 Training 요약:
      - ImageNet 데이터 셋으로 YOLO의 앞단 20개의 컨볼루션 계층을 사전 훈련시킨다.
      - 사전 훈련된 20개의 컨볼루션 계층 뒤에 4개의 컨볼루션 계층 및 2개의 전결합 계층을 추가한다.
      - YOLO 신경망의 마지막 계층에는 선형 활성화 함수(linear activation function)를 적용하고,
        나머지 모든 계층에는 leaky ReLU를 적용한다.
      - 구조상 문제 해결을 위해 아래 3가지 개선안을 적용한다.
        1) localization loss와 classification loss 중 localization loss의
           가중치를 증가시킨다.
        2) 객체가 없는 그리드 셀의 confidence loss보다 객체가 존재하는 그리드 셀의
           confidence loss의 가중치를 증가시킨다.
        3) bounding box의 너비(widht)와 높이(hegith)에 square root를 취해준 값을
           loss function으로 사용한다.
      - 과적합(overfitting)을 막기 위해 드롭아웃(dropout)과 data augmentation을 적용한다
---


### 3.3 Inference

YOLO의 그리드 디자인(gird design)은 하나의 객체를 여러 그리드 셀이 동시에 검출하는 단점이 있을 수 있음. 객체의 크기가 크거나 객체가 그리드 셀 경계에 인접해 있는 경우, 그 객체에 대한 bounding box가 여러개 생길 수 있음. 즉 하나의 그리드 셀이 아닌 여러 그리드 셀에서 해당 객체에 대한 bounding box를 예측할 수 있다는 뜻임. 이를 다중 검출(multiple detections) 문제라고 함. 이런 다중검출 문제는 비 최대 억제(non-maximal suppression)라는 방법을 통해 개선할 수 있음. YOLO는 비 최대 억제를 통해 mAP를 2~3%가량 향상시킴.

### 3.4 limitations of YOLO

---
    3. 4. Limitations of YOLO 요약: 
     - 작은 객체들이 몰려있는 경우 검출을 잘 못한다.
     - 훈련 단계에서 학습하지 못한 종횡비(aspect ratio)를 테스트 단계에서 마주치면 고전한다.
     - 큰 bounding box와 작은 bounding box의 loss에 대해 동일한 가중치를 둔다.
---

# 4. Comparison to Other Detection Systems

**Deformable parts models(DPM)**

객체 검출 모델 중 하나인 DPM은 슬라이딩 윈도(sliding window)방식을 사용함. DPM은 하나로 연결된 파이프라인이 아니라 서로 분리된 파이프라인으로 구성되어 있음. 독립적인 파이프라인이 각각 특징 추출(feature extraction), 위치 파악(region classification), bounding box 예측(bounding box prediction)등을 수행함. 반면 YOLO는 이렇게 분리된 파이프라인을 하나의 컨볼루션 신경망으로 대체한 모델임. 이 신경망은 bounding box 예측, 비 최대 억제 등을 한번에 처리함. 따라서 YOLO는 DPM보다 더 빠르고 정확함.

**R-CNN**

R-CNN은 슬라이딩 윈도 대신 region proposal 방식을 사용하여 객체를 검출하는 모델임. selective search라는 방식으로 여러 bounding box를 생성하고, 컨볼루션 신경망으로 feature를 추출하고, SVM으로 bounding box에 대한 점수를 측정함. 그리고 선형 모델(linear model)로 bounding box를 조정하고, 비 최대 억제(non-max suppression)로 중복된 검출을 제거함. 이 복잡한 파이프라인을 각 단계별로 독립적으로 튜닝해야 하기 때문에 R-CNN은 속도가 굉장히 느림. 정확성은 높지만 속도가 너무 느려 실시간 객체 검출 모델로 사용하기에는 한계가 있음.

# 5. Experiments

![](https://i.imgur.com/4H0KDBs.png)

## 5.1 Comparision to Other Real-Time Systems

---
    5. 1. Comparison to Other Real-Time Systems 요약:
      - YOLO는 기존 모델에 비해 속도는 월등히 빠르고, 정확도도 꽤 높은 수준이다.
---

## 5.2 VOC 2007 Error Analysis

![](https://i.imgur.com/brZYzxe.png)


## 5.3 Combining Fast R-CNN and YOLO

![](https://i.imgur.com/Fnib7d8.png)


## 5.4 VOC 2012 Results

![](https://i.imgur.com/fznI1Im.png)

## 5.5 Generalizatbility : Person Detection in Artwork

![](https://i.imgur.com/qkSFEZD.png)

---
    5. 5. Generalizability: Person Detection in Artwork 요약:
      - YOLO는 훈련 단계에서 접하지 못한 새로운 이미지도 잘 검출한다.
---

# 6. Real-Time Detection In the Wild

![](https://i.imgur.com/ZKMD5tV.jpg)

# 7. Conclusion

YOLO는 단순하면서도 빠르고 정확함. 또한 YOLO는 훈련 단계에서 보지 못한 새로운 이미지에 대해서도 객체를 잘 검출함. 즉, 새로운 이미지에 대해서도 강건함으로 애플리케이션에도 출분히 활용할만한 가치가 있음.



---

참고자료 

https://bkshin.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-YOLOYou-Only-Look-Once

https://curt-park.github.io/2017-03-26/yolo/

https://arxiv.org/pdf/1506.02640.pdf

https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.p
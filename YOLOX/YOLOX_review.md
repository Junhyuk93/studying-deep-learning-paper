# [논문 리뷰] YOLOX 논문 읽어보기🙄

![](https://i.imgur.com/nJuN1wU.png)


## Abstract

기존 YOLO detector을 바꿔 Anchor-free 방식을 적용하였고, Decoupled head와 SimOTA와 같은 다양한 Detection 기술을 전략하여 SOTA를 달성한 모델입니다.

- Anchor-free ?
    - 앵커 박스를 사용하지 않는 Achor-free detection (관련 논문 CircleNet: Anchor-free Detection with Circle Representation 읽어보기)


## Introduction

- object detection의 발전에서 YOLO 시리즈는 항상 실시간에서의 최적의 속도와 성능의 Trade-off를 가지고 있었습니다. 
- 뿐만 아니라, 2년간의 object detection의 Anchor Free Detector, Adavanced Label Assignment Strategies, End-to-end(NMS-free) detector등 다수의 기법을 연구하였습니다.
- YOLOv4와 YOLOv5의 파이프라인은 anchor-based detector에 최적화되어 있기 때문에, YOLOv3-SPP 모델을 base로 삼았습니다.
- Baseline에 비해서 AP가 향상되었으며, YOLOv3보다 높은 성능을 나타냅니다.

## YOLOX

- YOLOv3의 Darknet53을 baseline으로 설정했습니다.
- YOLOX는 1 Stage Detector로 Input - Backbone - Neck - Dense Predction의 구조를 가지고 Darknet53의 Backbone을 통해 Feature Map을 추출하며, SPP Layer를 통해 성능을 개선합니다.
    - SPP Layer (Spatial Pyramid Pooling Layer) : Convolutional layer가 임의의 사이즈로 입력을 취할 수 있게끔 해주는 Layer (입력 사이즈 제한을 제거함으로써 입력 이미지를 crop/wrap할 필요성을 없앤 것)
- FPN을 통해 Multi-Scale Feature Map을 얻고 이를 통해 작은 해상도의 Feature Map에서 큰 object를 추출하고 반대로 큰 해상도의 Feature Map에서는 작은 Object를 추출하는 Neck 구조를 사용했습니다.
- 최종적으로 Head 부분에서는 기존 YOLOv3와 다르게 Decoupled Head를 사용하였습니다.



### Decoupled Head

- YOLOv3에서는 하나의 Head에서 Classification과 Localization을 함께 진행하였으나(1-stage 방식), Classification과 Bbox Regression은 서로 다른 특성을 가진다는 내용이 연구되었습니다.
- Classification에는 Fully Connected Layer가 효과적이지만 Localization에서는 Convolution Head가 효율적이기 때문입니다.
- 따라서 이러한 Head 부분을 Double-Head 방식으로 변경하여 Classification에서는 Fully Connected Head를 적용하고, Localization에서는 Convolution Head를 적용함으로써 성능을 향상시켰습니다.
- Classification에서는 BCE Loss를 사용하였고 Localization에서는 IOU Loss를 사용하여 학습을 진행하였습니다.

![](https://i.imgur.com/KBLFvMp.png)

### Strong Augmentation

- 본 논문에서는 Mosaic Augmentation과 Mixup Augmentation을 적용하여 데이터를 증강하고 학습을 진행하였습니다.
- Strong data augmentation을 적용시켰을 때, ImageNet pre-trained 모델이 더 이상 이점을 가지지 못했기 때문에 Train from scratch 방식으로 학습을 하였습니다.

### Anchor-free

![](https://i.imgur.com/VU4bWT2.png)


- 최근 Anchor Free방식의 Detector들은 Achor방식의 방법론과 비교할 수 있을 만큼 성장하였습니다.
- 기존 Anchor 기반의 Detector들은 비록 그 성능이 뛰어날 수 있으나, 개발자들이 직접 Heuristic하게 Tuning을 진행해주어야 하는 복잡함이 존재했습니다. 또한 그렇게 Tuning된 Anchor Size 또한 특정 Task에 종속적이기에 General한 성능은 감소하는 이슈가 존재하였습니다.
- Anchor Free방식은 학습을 보다 간편하게 해주고, 다양한 Hyperparameter를 Tuning해야 하는 필요성이 없으며, 그로 인해 다양한 Task에 General하게 일정한 성능을 보장합니다.

### Multi Positive

![](https://i.imgur.com/rqHFVDU.png)


- 기존 YOLOv3의 Assigning Rule을 그대로 유지한다면 Anchor Free Version에서도 마찬가지로 중앙 위치 값 1개 만을 Positive Sample로 지정해야 하지만, 이는 그 주변 예측값을 모두 무시하게 되는 효과를 가집니다.
- 따라서 Positive Sample을 중앙 위치 값 주변 3x3 Size로 모두 지정함으로써 이러한 high quality 예측값에 대해서 이점을 취할 수 있습니다. (FCOS의 Center Sampling 기법)
- 이렇게 positive Sample을 증강시킴으로써, class imbalance도 상쇄시킬 수 있습니다.

### SimOTA

- Object Detection에서의 label Assignment는 각 지점에 대하여 Positive와 Negative를 할당해주는 업그레이드된 label Assign 전략을 사용하였습니다.
- Anchor Free 방식은 GT 박스 중앙 부분을 Positive로 처리하는데, 문제는 하나의 지점이 다수의 박수 내부에 존재할 때이고 이런 경우 단순히 point by point가 아닌 Global Labeling이 필요한데, 이를 최적화하는 방식으로 저자는 SimOTA를 적용하였습니다.
- OTA(Optimal Transportation Algorithm)은 Sinkhorn-knopp iteration등의 방법을 통해서 최적의 값을 찾아내는데 사용되는데, 이러한 iteration으로 인해 약 25%의 추가 학습 연산이 필요로 하게 됩니다.
- 약 300Epoch의 학습이 필요한 YOLOX에게 있어서 이는 꽤나 큰 Overhead이므로, 저자들은 이를 간단하게 iteration없이 수행하는 Simple OTA(SimOTA)를 적용하였으며 AP 45.0%를 47.3%로 향상시키는 효과가 있었다고 말합니다.


## Result

![](https://i.imgur.com/JCbwWIx.png)

- 기존 YOLO 모델들과 마찬가지로 속도와 성능간의 Trade Off가 존재하지만, 다른 모델들과 비교했을 때 높은 성능과 FPS를 나타냅니다.

## Conclusion

- 최신 Object Detection에서 발전된 기법들을 YOLOv3에 적용시키고 업그레이드한 버전이 YOLOX라고 생각합니다.
- Anchor Free 방식을 적용하여 기존 대비 General한 성능을 보장하고 모델 엔지니어로 하여금 Anchor와 관련된 다양한 Hyperparameter를 Tuning할 필요없이 학습을 가능하도록 하게 큰 장점이라고 생각됩니다.
- Decoupled Head와 Multi-Positive, SimOTA에 대해서 공부하고 다시 읽어보는게 더 쉽게 이해될 것 같습니다.


**참고자료**

---

SNU AI스터디 YOLOX 김진희님
[[논문 리뷰] YOLOX: EXCEEDING YOLO SERIES IN 2021](https://cryptosalamander.tistory.com/164)

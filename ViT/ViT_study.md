# VIT

## RNN to Transformer in NLP

- RNN은 오랜시간동안 NLP 분야에서 기본 모델로써 활용됨
- Transformer의 등장 이후 Transformer 중심으로 연구가 진행됨

![](https://i.imgur.com/RltkRPP.png)

## Transformer in Computer Vision

- **예전에는 CNN에 self attention을 적용시키는 방법을 고민함🤔**
    
    - Non-local neural networks
    ![](https://i.imgur.com/tRu4EQo.png)

    - Stand-alone self-attention in vision models
    ![](https://i.imgur.com/UJbUvls.png)

    - Axial-DeepLab
    ![](https://i.imgur.com/cPdqlcj.png)


- **사고방식의 전환으로 Transformer 모델 자체를 이용해 Vision task를 풀어보자!😮**

    - Vision Transformer
    - Data efficient image Transformer
    - TransGAN (??)

## Tansformer and Self Attention
- Transformer : **Attention** 만을 활용해 모델 구축
- Transformer의 핵심 아이디어 

## Seq2seq
- Seq2seq : 문장을 입력으로 받아 문장을 출력하는 모델 / 기계번역에 주로 사용
- Context vector: Decoder에게 전달되는 입력 문장의 정보
- Context vector의 크기가 제한적이기 때문에 입력 문장의 모든 정보를 전하기 어려움!!

![](https://i.imgur.com/JpPqA6H.png)

## Seq2seq with Attention
- Decoder가 특정 시점 단어를 출력할 때 encoder 정보 중 연관성이 있는 정보를 직접 선택!!


![](https://i.imgur.com/rYU7LWw.png)


decoder의 첫번째 시점과 encdoer 각 시점의 유사도를 계산하여 이것을 weight로 활용하고 이를 바탕으로 encoder 각 시점의 정보들을 통합하여 Context Vector를 만듬! 

Context Vector를 decoder의 첫번째 시점 단어를 출력할때 활용하여 좀 더 정확한 출력값을 내뱉을 수 있게 됨!

![](https://i.imgur.com/FxFaOJi.png)

## Attention vs Self Attention
- Attention (Decoder  Query / Encoder → Key, Value) / encoder, decoder 사이의 상관관계를 바탕으로 특징 추출
- Self attention (입력 데이터 → Query, Key, Value) / 데이터 내의 상관관계를 바탕으로 특징 추출

![](https://i.imgur.com/nn15dsp.png)


## Transformer vs CNN

- CNN : 이미지 전체의 정보를 통합하기 위해서는 몇 개의 layer 통과
- Transformer : 하나의 alyer로 전체 이미지 정보 통합 가능

![](https://i.imgur.com/EPOuhxx.png)

## Inductive bias

- Inductive bias : 새로운 데이터에 대해 좋은 성능을 내기 위해 모델에 사전적으로 주어지는 가정
- SVM : Margin 최대화 / CNN : 지역적인 정보 / RNN : 순차적인 정보

![](https://i.imgur.com/Z30H2eH.png)

- Transformer
    1차원 벡터로 만든 후 self attention (2차원의 지역적인 정보 유지 X)
    Weight이 input에 따라 유동적으로 변함
    
- CNN
    2차원의 지역적인 특성 유지
    학습 후 weight 고정됨!
    
<center>Transformer : inductive bias ↓ , 모델의 자유도 ↑</center>

![](https://i.imgur.com/6kIdyZT.png)

- **ViT의 단점 : 이미지의 2차원적 정보를 유지하지 못해 inductive bias를 학습하는 것이 어렵고 이에 따라 많은 양의 데이터를 필요로 함!!**

## VIT

### Abstract

- CNN을 사용할 필요 없이 image를 sequence of patches로 직접 사용하는 transformer 모델 자체가  classification에서 좋은 성능을 보인다는 것! 😮
- SOTA의 CNN 기반 모델과 비슷한 성능

![](https://i.imgur.com/GhfNwIA.png)


### Introduction

- image를 잘라서 patch(treated the same way as tokens (words))로 만들고 sequence를 linear embedding으로 만들어 transfoer에 넣었음!!

![](https://i.imgur.com/hyFW18t.png)

input image 가 전체 문장이고, image patch가 문장을 이루는 각각의 단어라고 이해하면 편함!


![](https://i.imgur.com/jVg8t7G.png)


- ① Classification token : classification을 위해 사용되는 token (BERT [CLS] token)
- ② Position embedding : patch의 위치 정보
- ①,② 는 학습을 통해 결정됨!
- (Classification token, Patch embedding) + Positional embedding = Transformer encoder 입력



![](https://i.imgur.com/oipI8bF.png)

- "Vanilla" Transformer encoder vs "ViT" Transformer
- Layer normealization의 위치가 Transformer 학습에 중요한 역할을 끼침(Learning Deep Trasformer Models for Machine Translation)
- 따라서 ViT 는 수정된 Transformer encoder를 적용함! (Normalization을 먼저 적용)

### Transformer encdoer : Self attention

![](https://i.imgur.com/FSyBC7B.png)

- Layer Normalization은 instance 단위로 nomalization 한다고 생각하면 됨!

![](https://i.imgur.com/Y4XQkKp.png)

- Transformer encoder: Self attention
- Encoder의 입력(z) → query, key, value 벡터 / W matrix: 학습되는 파라미터

실제로 학습이 일어나는건 W matrix을 통해 attention이 훈련됨.

![](https://i.imgur.com/qqgHcN7.png)

- query, key, value 에 동일한 W matrix가 곱해지고 Q,K,V가 계산됨!

- 각각의 query, key가 dot product를 통해 유사도를 계산하고 softmax 함수를 통해 0~1 사이의 값의 attention score를 계산함!

- value값과 attention score를 곱해주고 더해줌으로써 최종 attention의 output을 계산함!


### Transformer encoder : Multi-Head Self attention

- ViT에서는 Self attention 12번 수행

![](https://i.imgur.com/QocnowL.png)


### Transformer encoder : MLP

![](https://i.imgur.com/sDT09Ay.png)

- MLP에서는 2단으로 활성화 함수인 GELU를 사용하고 있음! 
- GELU가 뭐야?🤨 → [GELU](https://arxiv.org/abs/1606.08415)

### Transformer output

![](https://i.imgur.com/SiqDtSp.png)
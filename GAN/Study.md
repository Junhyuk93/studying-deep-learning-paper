# Generative Adversarial Nets (GAN) 논문 리뷰

## 1. Abstract

우리는 경쟁하는 과정을 통해 generative model을 추정하는 새로운 프레임워크를 제안합니다. 2개의 모델(generative model G, Discriminative model D)

- **Generative model(생성 모델) G** : Discriminative model이 구별할 수 없도록 traning data의 분포를 모사합니다.

- **Discriminative model(판별 모델) D** : sample 데이터가 G로부터 나온 데이터가 아닌 실제 training data로부터 나온 데이터일 확률을 추정합니다.

    - G를 학습하는 과정으로는 D가 sample 데이터 G로 부터 생성된 가짜 데이터와 실제 training 데이터를 판별하는데 어렵게 확률을 최대화 하는 것입니다.
    - 이 논문에서는 이와 같은 프레임워크를 minimax two-player game으로 표현합니다.
    - 임의의 함수 G, D의 공간에서 G가 training 데이터 분포를 모사하게 되면서 D가 실제 trarining 데이터인지, 생성된 가짜 데이터인지 판별하는 확률은 1/2가 됩니다. (실제 데이터와 G가 생성해내는 데이터의 판별이 어려워짐)
    - G와 D가 Multi-Layer Perceptrions으로 정의된 경우, 전체 시스템은 역전파를 통해 학습됩니다.


### 요약

이 논문에서는 GAN이라는 새로운 프레임워크를 제안하고 있으며, 이는 생성 모델과 판별 모델, G와 D 두가지 모델을 학습하며 G는 실제 training data의 분포를 모사하며 그와 비슷한 데이터를 생성하려하고, D는 실제 데이터와 G가 생성해낸 데이터를 구별하려하는 경쟁적인 구도를 갖추고 있습니다.

## 2. Introduction

- 딥러닝이 작동하는 방식은 인공지능 영역에서 마주하는 데이터의 종류에 대해서 모집단에 근사하는 확률 분포를 나타내는 계층 모델을 발견하는 것입니다. 지금까지는 고차원의 방대한 센싱 데이터를 클래스 레이블에 맵핑해서 구분하는 모델을 사용했습니다. (well-behaved gradient를 갖는 선형 활성화 함수들을 사용한 backpropagation, dropout 알고리즘 기반)
-  Deep generative model들은 maximum likelihood estimation과 관련된 전략들에서 발생하는 많은 확률 연산들을 근사하는 데 발생하는 어려움과 generative context에서는 앞서 모델 사용의 큰 성공을 이끌었던 선형 활성화 함수들의 이점들을 가져오는 것의 어려움이 있었기 때문에 큰 영향을 주진 못했습니다. 이 논문에서 소개될 새로운 generative model은 이러한 어려움을 극복했습니다.
- 이 논문에서 소개되는 adversarial nets 프레임워크의 컨셉은 **경쟁**으로 discriminative model은 sample data가 G model이 생성해낸 sample data인지, 실제 training data distribution인지 판별하는 것을 학습합니다.
- GAN의 경쟁하는 과정을 경찰(분류 모델, 판별자)과 위조지폐범(생성 모델, 생성자) 사이의 경쟁으로 비유하면, 위조지폐범은 최대한 진짜 같은 화폐를 만들어 경찰을 속이기 위해 노력하고, 경찰은 진짜 화폐와 가짜 화폐를 완벽히 판별하여 위조지폐범을 검거하는 것을 목표로 한다. 이러한 경쟁하는 과정의 반복은 어느 순간 위조 지폐범이 진짜와 다를 바 없는 위조지폐를 만들 수 있고 경찰이 위조 지폐를 구별할 수 있는 확률 역시 50%로 수렴하게 됨으로써 경찰이 위조 지폐와 실제 화폐를 구분할 수 없는 상태에 이르도록 합니다.


![](https://i.imgur.com/gjk4rkD.png)

- 이 프레임워크는 많은 특별한 학습 알고리즘들과 optimization 알고리즘을 사용할 수 있습니다.
- abstract에서 나왔듯이 이 논문에서는 Multi-layer perceptron을 사용하면 다른 복잡한 네트워크 없이 오직 forward propagation/back propagation/dropout algorithm으로 학습이 가능합니다.

### 요약

결국 GAN의 핵심 컨ㅅ넵은 각각의 역할을 가진 두 모델을 통해 적대적 학습을 하면서 '진짜같은 가짜'를 생성해내는 능력을 키워주는 것



## 3. Adversarial nets

- adversarial modeling 프레임워크는 앞서 말했듯이 가장 간단하므로, multi layer perceptrons 모델을 적용합니다.
- 학습 초반에는 G가 생성해내는 image는 D가 실제 데이터의 샘플인지, 생성해낸 가짜 샘플인지 바로 구별 할 수 있을 만큼 낮은 성능이기 때문에 D(G(z))의 결과가 0에 가깝습니다. 즉 z로 부터 G가 생성해낸 이미지가 D가 판별하였을 때 바로 가짜라고 판별할 수 있다고 하는 것을 수식으로 표현한 것입니다. 학습이 진행될수록, G는 실제 데이터의 분포를 모사하면서 D(G(z))의 값이 1이 되도록 발전합니다. 이는 G가 생성해낸 이미지가 D가 판별하였을 때 진짜라고 판별해버리는 것을 표현한 것입니다.

![](https://i.imgur.com/xMx8uS2.png)

- 왼쪽 항 : 실제 데이터 x를 discriminator에 넣었을 때 나오는 결과를 log로 취했을 때 얻는 기댓값
- 오른쪽 항 : 가짜 데이터 z를 generator에 넣었을 때 나오는 결과를 discriminator에 넣었을 때 그 결과를 log(1-result)로 취해주고 얻는 기댓값

![](https://i.imgur.com/Nc0ba2u.png)

- 이 방정식을 D의 입장, G의 입장에서 각각 이해해본다면, D의 입장에서 value function V(D,G)의 이상적인 결과를 생각해봤을 때, D가 매우 뛰어난 성능으로 판별을 잘 해낸다고 가정하면 D가 판별하려는 데이터가 실제 데이터에서 온 샘플일 경우에는 D(x)가 1이 되어 첫번째 항은 0이 되어 사라지고 G(z)가 생성해낸 가짜 이미지를 구별해낼 수 있으므로 D(G(z))는 =이 되어 두번째 항 log(1-0) = log1 = 0이 성립해 V(D,G) = 0이 됩니다. 즉 D의 입장에서 얻을 수 있는 이상적인 결과, '최댓값'은 0임을 확인할 수 있습니다.
- G의 입장에서 이 value function V(D,G)의 이상적인 결과를 생각해보면, G가 D에서 구별하지 못할 만큼 진짜와 같은 데이터를 잘 생성해낸다고 했을 때, 첫번째 항은 D가 구별해내는 것에 대한 항으로 G의 성능에 의해 결정될 수 있는 항이 아니므로 넘어가고 두번째 항을 살펴보면 G가 생성해낸 데이터는 D를 속일 수 있는 성능이라 가정했기 때문에 D는 G가 생성해낸 이미지를 가짜라고 인식하지 못하고 진짜라고 결정내버립니다. 그러므로 D(G(z)) = 1이 되고 log(1-1)= log0 = 마이너스 무한대 가 됩니다. 즉 G의 입장에서 얻을 수 있는 이상적인 결과, '최솟값'은 '마이너스 무한대'임을 확인할 수 있습니다.
- 다시말해, D는 training data의 sample과 G의 sample에 진짜인지 가짜인지 올바른 라벨을 지정할 확률을 최대화하기 위해 학습하고, G는 log(1-D(G(z))를 최소화(D(G(z))를 최대화)하기 위해 학습 되는 것입니다.
- D 입장에서는 V(D,G)를 최대화시키려고, G 입장에서는 V(D,G)를 최소화시키려고 하고, 논문에서는 D와 G를 V(D,G)를 갖는 two-player minmax game 으로 표현합니다.

## 4. Theoritical Result

G는 z~p_z일때 얻어지는 샘플들의 확률분포 G(z)로써 p_g를 암묵적으로 정의합니다. 그러므로 아래에서 살펴보는 알고리즘 1 p_data에 대한 좋은 estimator로 수렴되길 원합니다.

![](https://i.imgur.com/YpL5Ktf.png)

아래 과정을 k번 반복합니다. (paper에선 k=1 로 실험)

1. m개의 노이즈 샘플을 p_g(z)로부터 샘플링
2. m개의 실제 데이터 샘플을 p_data(x)로부터 샘플링
3. 경사상승법을 이용해 V(G,D)식 전체를 최대화하도록 discriminator 파라미터 업데이트

이후

1. m개의 노이즈 샘플을 p_g(z)로부터 샘플링
2. V(G,D)에서 log(1-D(G(z)))를 최소화 하도록 generator 파라미터 업데이트

## 4.1 global optimality of p_g = p_data

모든 가능한 G에 대해 최적의 discriminator D를 구해봅시다.

![](https://i.imgur.com/JJa7OHc.png)

위의 식을 D(x)에 대해 편미분하고 결과값을 0이라고 가정하면 optimal한 D는 아래와 같은 결과값을 나타냅니다.

![](https://i.imgur.com/M0mg5gk.png)

이렇게 얻은 optimal D를 원래의 목적함수 식에 대입하여 생성기 G에 대한 Virtual Traning Criterion C(G)를 다음과 같이 유도할 수 있습니다.

![](https://i.imgur.com/TNDBo7f.png)

위의 C(G)는 generator가 최소화하고자 하는 기준이 되며, 이것은 global minimum은 오직 p_g = p_data일때 달성됩니다. 그 점에서 C(G)값은 log4 가 됩니다.

## 4.2 Convergence of Algorithm 1

G와 D가 충분한 capacity를 가지며, algorithm 1의 각 step에서 discriminator가 주어진 G에 대해 최적점에 도달하는게 가능함과 동시에 p_g가 위에서 제시한 criterion을 향상시키도록 업데이트 되는 한, p_g는 p_data에 수렴합니다.

- 그런데 실질적으로 adversarial nets은 함수 G(z;θ_g)를 통하여 분포 p_g의 제한된 family 만을 나타나게 되며, 우리가 수행하는 최적화는 사실 p_g를 직접 최적화 하는게 아닌 θ_g를 최적화하는 것입니다. 그래서 위의 증명이 적용되지 않습니다.

그러나 실무에서 MLP가 보여주는 훌륭한 퍼포먼스는 위와 같은 이론적 gurantee의 부족에도 불구하고 사용할 수 있는 합리적인 모델이라는 사실을 말해줍니다.

## 5. Experiment

- generator에서는 ReLU, sigmoid activation을 섞어 사용했습니다.
- discriminator에서는 maxout activation만을 사용했습니다.
- discriminator 훈련시 dropout을 사용했습니다.
- 저자들이 제안하는 프레임워크는 generator의 중간 레이어들에 dropout과 noise 추가를 이론적으로 허용하지만, 오직 generator의 최하단 레이어에만 노이즈를 추가했다고 합니다.

## 6. Advantages and disadvantages

#### 장점 : 
- Markov chain이 불필요합니다.
- 학습 단계에서 inference가 필요하지 않습니다.
- 모델에 다양한 함수들이 통합될 수 있습니다.
- generator network이 데이터로부터 직접적으로 업데이트 되지 않고 오직 discriminator로 부터 흘러들어오는 gradient만을 이용해 학습될 수 있습니다. (이는 input의 요소들이 직접적으로 생성기의 파라미터에 복사되지 않는다는 걸 의미합니다)
- MC 기반 방법들은 체인이 mode들 간에 혼합될 수 있도록 하는 과정에서 분포가 다소 blurry해지는 경향이 있는 반면 GAN은 sharp한 표현을 얻습니다.

#### 단점 : 
- p_g(x)에 대한 명시적인 표현이 없습니다
- 훈련동안 D와 G가 반드시 균형을 잘 맞춰 학습되어야 합니다
- 최적해로의 수렴에 있어 이론적 보장이 부족합니다

## 7. Conclusions and future work

- 클래스 레이블 c를 생성기와 판별기에 모두 추가하는 것으로 조건부 생성모델 p(x|c)를 얻을 수 있습니다.
- x가 주어졌을 때 z를 예측하는 보조 네트워크를 훈련시켜 학습된 근사추론을 진행할 수 있습니다.
- 파라미터들을 공유하는 조건부 모델들의 family를 훈련시킴으로써 모든 조건부 확률 p(x_s|x_nots), (s : x의 인덱스들의 부분집합)을 근사적으로 모델링할 수 있습니다.
- 준지도학습 : discriminator에 의해 얻어지는 중간 단계 feature들은 레이블이 일부만 있는 데이터셋을 다룰 때 분류기의 성능을 향상시킬 수 있습니다.
- 효율성 개선 : G와 D의 조정을 위한 더 좋은 방법을 고안하거나 훈련동안 z를 샘플링 하기위해 더 나은 분포를 결정함으로써 학습을 가속화할 수 있습니다.
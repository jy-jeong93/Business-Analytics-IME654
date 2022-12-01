## Ensemble Learning with Python Tutorial

#### --Ensemble Learning 설명 ()


이번 튜토리얼에서는 ensemble learning을 소개하고 정형 데이터셋을 통해서 다양한 앙상블 기법을 적용해보는 것을 목표로 한다.

# Ensemble Learning 방법론
Ensemble Learning이란 여러 모델들을 함께 사용하여 기존보다 성능을 더 올리는 방법을 말한다.
다음과 같이 크게 세 가지 갈래로 나누어 생각할 수 있다.

1. Bagging
Bootstrap Aggregating의 약자이며 bootstrap을 이용하는 방법이다. 주어진 데이터셋에 대해 bootstrap 샘플링을 이용하여 단일 알고리즘 모델보다 더 좋은 모형을 만들 수 있는 기법이다.

* Bootstrap: 주어진 데이터셋에서 random sampling을 거쳐 새로운 데이터셋을 만들어내는 과정이다.
![image](https://user-images.githubusercontent.com/115562646/205045270-afc769ae-0d0c-4375-9784-a6e8ec80b04c.png)
각 데이터셋은 복원추출을 통해 기존 데이터셋만큼의 크기를 갖도록 샘플링된다. 상단 이미지 예시와 같이 개별 샘플링된 데이터셋을 bootstrap이라고 한다.
* ex) Random Forest


2. Voting
voting은 크게 Hard voting과 soft voting으로 나눌 수 있다.
* Hard voting: 각 하위 학습 모델(weak learner)들의 예측 결과값을 바탕으로 다수결 투표를 하는 방식
* Soft Voting: 각 하위 학습 모델(weak learner)들의 예측 확률값의 평균 또는 가중치 합을 사용하는 방식


3. Boosting
모델 iteration의 결과에 따라 데이터셋 샘플에 대한 가중치를 부여하며 모델을 업데이트하는 방식이다. 모델을 반복할 때마다 각 샘플의 중요도에 따라 다른 분류기가 만들어지고 최종적으로 모든 iteration에서 생성된 모델의 결과를 voting한다.
* ex) Adaptive Boosting(AdaBoost)와 Gradient Boosting Model(GBM) 계열로 나눌 수 있다.



# No Free Lunch Theorem?
머신러닝은 다양한 샘플 데이터에 학습(fitting)을 시킴으로써 일반화되는 것을 목적으로 둔다. 특수한 부분에서 출발하여 일반성을 확보하고자 하는 논리로 본다면 귀납적 추론으로 머신러닝을 바라볼 수 있겠다. 그렇다면 '모델이 학습을 한다'라는 의미를 생각해보면, 이는 샘플 데이터로 구성된 가설 공간속에서 데이터에 알맞은 가설을 채택하는 의미라고 생각할 수 있겠다. 그러나 다양한 가설이 존재하는 만큼 귀납적 편향 문제에 마주한다.

다음 이미지는 다양한 데이터셋에 대한 각 알고리즘들 성능 비교 예시이다. (https://www.kdnuggets.com/2019/09/no-free-lunch-data-science.html)
![image](https://user-images.githubusercontent.com/115562646/205047080-c6e31605-702b-4350-b77a-ec5f9b663bde.png)






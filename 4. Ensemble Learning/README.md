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

그래서 어떤 알고리즘(모델)이 가장 우월한지, 더 일반성을 확보한 모델은 무엇인지에 대해서 생각해보면 위의 이미지와 같이 그런 알고리즘은 없다고 할 수 있다.
성능이 좋다고하는 알고리즘도 어떤 데이터셋이 주어지느냐에 따라서 성능이 달라진다. No Free Lunch Theorem의 요지는 '모든 문제에 대하여 다른 모델보다 항상 우월한 모델은 없다'이다.
학습이 된 모델은 주어진 데이터에 대해서 걸맞는 패턴을 학습하고 적응을 했다고 볼 수 있기 때문에 어떤 데이터셋(어떤 문제)에 대해서도 항상 우월한 모델은 없다고 생각한다.



# Bias-Variance Decomposition
특정 데이터에 대한 오차를 편향과 분산에 의한 에러로 나눌 수 있다. 편향(Bias)은 정답과 평균 추정치 차이고 분산(Variance)은 평균 추정치와 특정 데이터셋에 대한 추정치 차이를 의미한다. 편향이 높으면 과소적합이 발생하며, 분산이 높으면 과적합이 발생한다.

아래는 편향과 분산에 대한 수식 및 예시 이미지이다.
![image](https://user-images.githubusercontent.com/115562646/205048472-3fa62c02-d547-4182-887d-80c98715f71c.png)
편향과 분산으로 나누어서 앙상블 기법을 생각할 수 있다.

# Ensemble learning의 목적
앙상블의 목적은 각 단일 모델의 좋은 성능을 유지시키면서 다양성(diversity)을 확보하는데 있다.
* Implicit diversity : 전체 데이터셋의 부분집합에 해당하는 여러 데이터셋을 준비한 뒤 따로 학습
* Explicit diversity : 먼저 생성된 모델의 측정값으로부터 새로운 모델을 생성하여 학습

Low Bias, High Variance 단일 모델: Decision Tree, ANN, SVM, k값이 작은 K-NN 등 이러한 모델에 대해서는 Bagging이나 Random Forest 등을 통해서 분산을 줄이면 효과적일 것이다.
반대로 High Bias, Low Variance 단일 모델: Logistic Regression, k값이 큰 K-NN 등에 대해서는 Boosting을 통해서 편향을 줄이면 효과적일 것이다.


# Ensemble 기법을 활용한 MNIST 데이터 분석
MNIST 데이터셋은 28x28 gray scale이며 0~9까지의 숫자 이미지 데이터셋이다. 전체 데이터셋은 총 70000만장이며 이를 학습에는 60000만장, 테스트에는 10000장 활용할 것이다.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import torchvision.datasets as datasets

trainset = datasets.MNIST(root='./data', train=True, download=False)
testset = datasets.MNIST(root='./data', train=False, download=False)

X_train, y_train = trainset.data.numpy().reshape(-1,28*28), trainset.targets.numpy()
X_test, y_test = testset.data.numpy().reshape(-1,28*28), testset.targets.numpy()

X = {'train':X_train, 'test':X_test}
y = {'train':y_train, 'test':y_test}
```

### Hard voting과 Soft voting 비교
* Hard voting: 각 하위 학습 모델(weak learner)들의 예측 결과값을 바탕으로 다수결 투표를 하는 방식
* Soft Voting: 각 하위 학습 모델(weak learner)들의 예측 확률값의 평균 또는 가중치 합을 사용하는 방식
 
MNIST 분류 문제를 풀어보기 위해서 실험에서 사용할 하위 학습 모델은 Logistic regression, random forest(classification), support vector classification이다.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

log_clf = LogisticRegression(random_state=2022)
rnd_clf = RandomForestClassifier(random_state=2022)
svm_clf = SVC(random_state=2022)

hardvoting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
hardvoting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, hardvoting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

|Hard Voting|Logistic Regression|RandomForest|VotingClassifier|
|:---------:|:-----------------:|:----------:|:--------------:|
| Accuracy  |      0.9255       |   0.9690   |     0.9709     |


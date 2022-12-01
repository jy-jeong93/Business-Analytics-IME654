## Ensemble Learning with Python Tutorial

#### --Ensemble Learning 설명 ((https://github.com/jy-jeong93/Business-Analytics-IME654/blob/main/4.%20Ensemble%20Learning/%EB%B9%84%EC%A6%88%EB%8B%88%EC%8A%A4%EC%95%A0%EB%84%90%EB%A6%AC%ED%8B%B1%EC%8A%A4_ensemble_learning.pdf))


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

![image](https://user-images.githubusercontent.com/115562646/205064709-12d977e6-98e9-453d-9bc0-e11d223ea8cd.png)


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
 
MNIST 분류 문제를 풀어보기 위해서 voting에 사용할 하위 학습 모델은 Logistic regression, random forest(classification), support vector classification이다.

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
### Hard voting 결과
|Hard Voting|Logistic Regression|RandomForest|   SVC   |VotingClassifier|
|:---------:|:-----------------:|:----------:|:-------:|:--------------:|
| Accuracy  |      0.9255       |   0.9690   |  0.9782 |     0.9709     |



```python
log_clf = LogisticRegression(random_state=2022)
rnd_clf = RandomForestClassifier(random_state=2022)
svm_clf = SVC(probability=True, random_state=2022)

softvoting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='soft')
softvoting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, softvoting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```


### Soft voting 결과
|Soft Voting|Logistic Regression|RandomForest|   SVC   |VotingClassifier|
|:---------:|:-----------------:|:----------:|:-------:|:--------------:|
| Accuracy  |      0.9255       |   0.9690   |  0.9792 |     0.9718     |



Hard voting과 soft voting 결과를 비교해본 결과 SVC와 VotingClassifier 성능에서 soft voting 방식이 조금 더 좋았다.
두 Voting의 연산에 대해서 생각을 해보았을때는 soft voting이 더 정교한 확률을 구하는 방식이므로 좋을 것이라고 생각했었다.
그러나 ensemble 개념과 같이 어떤 voting이 더 좋다라고는 할 수 없을 것이다. 문제 상황과 데이터셋에 맞는 합리적인 voting방식을 생각하고 구현하는 것이 더 중요할 것이라 생각한다.



### 단일 Decision Tree와 Random Forest 비교
단일 의사결정나무 모델과 배깅을 적용한 의사결정나무의 앙상블인 랜덤 포레스트간 성능 차이를 비교한다. 또한 학습된 랜덤 포레스트를 통해서 이미지 데이터의 어떤 픽셀이 중요한지, 중요도를 반영하여 그래프로 나타내볼 것이다.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

tree_clf = DecisionTreeClassifier(random_state=2022)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print('DecisionTree Accuracy =', accuracy_score(y_test, y_pred_tree))

# Random Forest
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=2022)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
print('RandomForest Accuracy =', accuracy_score(y_test, y_pred_rf))
```


### 단일 의사결정나무와 랜덤포레스트 비교
|Soft Voting|Decision Tree|RandomForest|
|:---------:|:-----------:|:----------:|
| Accuracy  |    0.8793   |   0.8278   |


단일 의사결정나무가 랜덤포레스트보다 더 좋은 성능을 보였다. 
단일 의사결정나무에 배깅 기법을 적용하여 다양성을 더욱 확보한 랜덤포레스트가 더 낮은 성능을 보이는 것은 하이퍼파라미터때문이라고 생각한다. 
따라서 하이퍼파라미터 gridsearch를 수행하고 단일 의사결정나무와 성능을 재비교하였다.

Gridsearch
*n_setimators = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
*max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, None]
*min_samples_leaf = [1, 2, 4, 8, 16]
*min_samples_split = [1, 2, 4, 8, 16]

```python
from sklearn.model_selection import GridSearchCV

temp_list = list(range(1,10))
temp_list.append(None)
params = {
    'n_estimators' : tuple(range(50, 501, 50)),
    'max_depth' : tuple(temp_list),
    'min_samples_leaf' : tuple(map(lambda x: 2**x, range(5))),
    'min_samples_split' : tuple(map(lambda x: 2**x, range(5)))
}

rf_run = RandomForestClassifier(random_state=2022, n_jobs=-1)
grid_cv = GridSearchCV(rf_run, param_grid=params, cv=2, n_jobs=-1)
grid_cv.fit(X_train, y_train)

print('최적 하이퍼 파라미터:', grid_cv.best_params_)
print('최적 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))
```
```python
최적 하이퍼 파라미터: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
최적 예측 정확도: 0.9643
```


```python
rf_run = RandomForestClassifier(random_state=2022, 
                               max_depth=grid_cv.best_params_.get('max_depth'),
                               min_samples_leaf=grid_cv.best_params_.get('min_samples_leaf'),
                               min_samples_split=grid_cv.best_params_.get('min_samples_split'),
                               n_estimators=grid_cv.best_params_.get('n_estimators'),
                               )

rf_run.fit(X_train, y_train)
y_pred_rf = rf_run.predict(X_test)
print('RandomForest with grid search Accuracy =', accuracy_score(y_test, y_pred_rf))
```
```python
RandomForest with grid search Accuracy = 0.9709
```

### 단일 의사결정나무와 랜덤포레스트(grid search) 재비교
|Soft Voting|Decision Tree|RandomForest|RandomForest with grid search|
|:---------:|:-----------:|:----------:|:---------------------------:|
| Accuracy  |    0.8793   |   0.8278   |           0.9709            |

Grid search를 통해서 랜덤포레스트 하이퍼파라미터를 탐색하였고 정확도가 약 0.14 올랐다.
단일 의사결정나무와 비교를 하였을때도 랜덤포레스트가 배깅 기법을 적용한 만큼 압도적으로 좋은 성능을 보임을 확인하였다.


### 특성 중요도 확인
학습된 랜덤 포레스트를 통해서 이미지 데이터의 어떤 픽셀이 중요한지, 중요도를 반영한 이미지이다.
숫자를 나타내는 중앙 부근 픽셀들이 역시 중요한 픽셀이라고 학습됨을 확인했다.
![image](https://user-images.githubusercontent.com/115562646/205122369-8dd500e1-28bf-4834-9297-b05774b7cf16.png)





## 결론
이번 튜토리얼에서는 앙상블 기법에 대해서 MNIST 데이터셋에 대한 실험을 통해 알아보았다.
단일 알고리즘보다 모델 복잡도가 올라가는 만큼 학습 시간에 있어서 trade-off가 존재한다. 그러나 수업에서 배웠던 내용대로 단일 알고리즘보다 앙상블 알고리즘이 더 좋은 성능을 보임을 확인하였다.

## Support Vector Machines with Python Tutorial

#### --SVM 설명 ([ppt slide](https://github.com/jy-jeong93/Business-Analytics-IME654/blob/main/2.Kernel-based%20Learning/SVM_slide.pdf))--

이번 튜토리얼에서는 SVM을 소개하고 정형 데이터셋을 통해서 다양한 kernel과 SVM을 적용해보는 것을 목표로 한다.

### 튜토리얼 목차
 1. Support Vector Machines(SVM)이란?
 2. Linear SVMs - Hard Margin Classification
 3. Soft Margin Classification
 4. NonLinear SVM Classification with Kernel Functions




## 1. Support Vector Machines(SVM)이란?

Support vector machines(SVM)은 머신러닝 분야 중 하나로써 신호 처리, 의료 응용 분야, 자연어 처리, 음성 및 영상 인식 등 다양한 분야에서 여러 분류 및 회귀 문제에 사용되는 지도 학습 알고리즘이다. 이 알고리즘은 하나의 클래스에 대한 데이터 포인트들을 다른 클래스들의 데이터 포인트들과 최대한 잘 구분해내는 초평면(hyperplane)을 찾는 것을 목표로 한다.
아래 그림을 보면 데이터 포인트들을 나눌 수 있는 분류 경계선은 $2^n$이다.
<p align="center">
 
![image](https://user-images.githubusercontent.com/115562646/199581342-49bcf5c4-d833-49f2-bd87-4483e5d64ea7.png)
 
</p>


## 2. Linear SVMs - Hard Margin Classification

Linear SVM은 데이터 포인트들을 최대한 잘 구분해내는 선형분리를 찾는 것이 목적이며, 아래 그림과 같이 두 데이터의 클래스를 분리할 수 있는 수 많은 직선들 중 두 데이터 클래스간 간격(margin)이 최대가 되는 MMH(Maximum Marginal Hyperplane, 최대 마진 초평면)을 찾아 구분하는 방법이다.
![image](https://user-images.githubusercontent.com/115562646/199652997-789ca4a9-59c0-4a2c-ba9f-d587d687d217.png)
![image](https://user-images.githubusercontent.com/115562646/199655813-86c6ea08-e208-4033-9352-8013e36d60c4.png)

```python

import os
import numpy as np

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.svm import SVC
from sklearn import datasets
from sklearn.datasets import make_classification
sn.set()
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
matplotlib.rc('font', family='Malgun Gothic')  # Windows
plt.rcParams['axes.unicode_minus'] = False


plt.figure(figsize=(8, 8))
plt.title("두개의 독립변수 모두 클래스와 상관관계가 있는 가상데이터")
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=11111)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
            s=100, edgecolor="k", linewidth=2)

plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.show()

```
![image](https://user-images.githubusercontent.com/115562646/199671510-d6e2364e-dcfb-43e3-b857-98fe42fbbb91.png)


```python

# SVM 분류 모델
svm_clf = SVC(kernel="linear", C=float("inf"))
svm_clf.fit(X, y)

x0 = np.linspace(-5, 5.5, 200)
pred_1 = 5*x0 - 5
pred_2 = x0 - 1.8
pred_3 = 0.1 * x0 + 0.5

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # 결정 경계에서 w0*x0 + w1*x1 + b = 0 이므로
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_  # support vectors
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

plt.figure(figsize=(15,4))

plt.subplot(121)
plt.plot(x0, pred_1, "g--", linewidth=2)
plt.plot(x0, pred_2, "m-", linewidth=2)
plt.plot(x0, pred_3, "r-", linewidth=2)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="class0")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="class1")
plt.xlabel("feature1", fontsize=14)
plt.ylabel("feature2", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 5.5, 0, 2])

plt.subplot(122)
plot_svc_decision_boundary(svm_clf, -5, 5.5)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo")
plt.xlabel("feature1", fontsize=14)
plt.axis([-3, 5.5, 0, 2])

plt.show()

```

![image](https://user-images.githubusercontent.com/115562646/199672455-73ac1882-6d13-49da-afc4-13b7548d728e.png)

상단의 좌측 그림을 보면 생성된 데이터셋을 명확히 구분짓지 못하는 것을 볼 수 있다. 우측 그림의 SVM 방식은 Hard Margin SVM이라고 한다. 이 방식이 두 개의 클래스를 분리함에 있어서 모든 데이터 포인트가 분리 초평면을 기준으로 정확히 나눠주는 방식이다. 
하지만 데이터에 노이즈 또는 이상치, 두 클래스 사이의 overlap 등이 존재한다면 두 클래스를 엄격하게 분리하기는 힘들 것이다.
![image](https://user-images.githubusercontent.com/115562646/199673135-21aaf402-0b91-4224-b01f-a32e63bdac6a.png)


## 3. Soft Margin Classification

데이터에 노이즈 또는 이상치가 존재한다면 기존 Hard Margin SVM을 적용하긴 힘들 것이다. 이러한 한계점을 해결하기 위해 Soft Margin SVM이 개발되었다. Soft Margin SVM은 Hard Margin SVM의 support vector가 위치한 경계선에서 slack variable을 두어 오류를 일정 수준 허용해주는 방법이다.

이때 선형 분리를 하기 힘든 예제를 다시 생성하였고 이 데이터를 가지고 실험을 진행하였다.

![image](https://user-images.githubusercontent.com/115562646/199666949-26345b6f-9471-41a4-a6bc-e804cadcc406.png)
상단 그림의 수식에서 C는 hyperparameter인 slack variable이며 일종의 penalty라고 볼 수 있다.
* C가 커지면 오류를 허용하는 정도가 작아지며, 따라서 Margin이 작아진다.
* C가 작아지면 오류를 허용하는 정도가 커지며, 따라서 Margin이 커진다.

하단에서 C=1일때와 C=100일때 비교를 진행하겠다.


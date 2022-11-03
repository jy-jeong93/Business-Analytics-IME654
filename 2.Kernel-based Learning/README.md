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
```python

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

plt.figure(figsize=(8, 8))
plt.title("두개의 독립변수 모두 클래스와 상관관계가 있는 가상데이터")
X, y = make_classification(n_samples=290, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=220)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
            s=100, edgecolor="k", linewidth=2)

plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.show()

```
![image](https://user-images.githubusercontent.com/115562646/199673403-9a731ac8-39b4-49d9-b06c-8f999db352b5.png)



![image](https://user-images.githubusercontent.com/115562646/199666949-26345b6f-9471-41a4-a6bc-e804cadcc406.png)
상단 그림의 수식에서 C는 hyperparameter인 slack variable이며 일종의 penalty라고 볼 수 있다.

* C가 커지면 오류를 허용하는 정도가 작아지며, 따라서 Margin이 작아진다.
* C가 작아지면 오류를 허용하는 정도가 커지며, 따라서 Margin이 커진다.

하단에서 C=1일때와 C=100일때 비교를 진행하겠다.

```python 


scaler = StandardScaler()
svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)

scaled_svm_clf1 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf1),
    ])
scaled_svm_clf2 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf2),
    ])

scaled_svm_clf1.fit(X, y)
scaled_svm_clf2.fit(X, y)


b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])
w1 = svm_clf1.coef_[0] / scaler.scale_
w2 = svm_clf2.coef_[0] / scaler.scale_
svm_clf1.intercept_ = np.array([b1])
svm_clf2.intercept_ = np.array([b2])
svm_clf1.coef_ = np.array([w1])
svm_clf2.coef_ = np.array([w2])
t = y * 2 - 1
support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()
support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()
svm_clf1.support_vectors_ = X[support_vectors_idx1]
svm_clf2.support_vectors_ = X[support_vectors_idx2]


plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="class0")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="class1")
plot_svc_decision_boundary(svm_clf1, -10, 10)
plt.xlabel("feature1", fontsize=14)
plt.ylabel("feature2", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.title("$C = {}$".format(svm_clf1.C), fontsize=16)
plt.axis([-5, 5, -5, 5])

plt.subplot(122)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
plot_svc_decision_boundary(svm_clf2, -10, 10)
plt.xlabel("feature1", fontsize=14)
plt.title("$C = {}$".format(svm_clf2.C), fontsize=16)
plt.axis([-5, 5, -5, 5]);

```

![image](https://user-images.githubusercontent.com/115562646/199673554-fe546f3b-2c48-4a54-bbbe-ad61f6dec83c.png)

C=1일때는 허용해주는 오류 정도가 크기때문에 마진이 커진 것을 확인할 수 있고, C=100일 경우 허용해주는 오류 정도가 적기때문에 마진이 작아진 것을 확인할 수 있다.


## 4. NonLinear SVM classification
현실에는 선형적으로 분류할 수 없는 비선형성을 가진 데이터셋이 많다. 따라서 Hard Margin SVM과 Soft Margin SVM으로는 한계가 있을 것이다. 이러한 한계점들을 개선하기 위해 kernel 함수를 이용하여 비선형 데이터셋을 고차원으로 보내어 선형 분리가 가능하게끔 만들어주는 방법들이 있다.

![image](https://user-images.githubusercontent.com/115562646/199674110-f4aa27f8-7e0f-4295-adcd-a34864fee6d7.png)


![image](https://user-images.githubusercontent.com/115562646/199674345-a8cdc9c8-12c4-466c-9f36-230790b85de1.png)

![image](https://user-images.githubusercontent.com/115562646/199674390-197457d8-857e-4cb6-8694-fec2d1164d60.png)




아래에서는 sklearn에서 제공하는 make moons로 선형분류가 힘든 데이터셋을 임의로 만들었다. 만들어진 데이터셋에 Polynomial Kernel과 RBF Kernel을 적용할 것이다.
```python
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.15, random_state=2022)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()
```
![image](https://user-images.githubusercontent.com/115562646/199674465-e6e30663-3d37-433a-b6a7-26b7060e3411.png)


```python

from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVC(C=10, loss='hinge', random_state=42))
])

polynomial_svm_clf.fit(X, y)
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title('다항 특성을 사용한 Linear SVM 분류기')
plt.show()

from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=3, coef0=0.1, C=5))
])

poly_kernel_svm_clf.fit(X, y)


poly100_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=100, C=5))
    ])

poly100_kernel_svm_clf.fit(X, y)

plt.figure(figsize=(11, 4))

plt.subplot(121)
plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=3, r=0.1, C=5$", fontsize=18)

plt.subplot(122)
plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=3, r=100, C=5$", fontsize=18)

plt.show()
```

```python
gamma1, gamma2 = 0.1, 5
C1, C2 = 0.001, 1000
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clfs = []
for gamma, C in hyperparams:
    rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
        ])
    rbf_kernel_svm_clf.fit(X, y)
    svm_clfs.append(rbf_kernel_svm_clf)

plt.figure(figsize=(16, 15))

for i, svm_clf in enumerate(svm_clfs):
    plt.subplot(221 + i)
    plot_predictions(svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)

plt.show()
```

# Polynomial Kernel 적용시
![image](https://user-images.githubusercontent.com/115562646/199674639-4e45d3f9-8883-44ab-b075-b0809eb81252.png)


# RBF Kernel 적용시
![image](https://user-images.githubusercontent.com/115562646/199674755-47eca25d-0cb1-4d3a-bb7b-a1e23e513ed7.png)

## Anomaly Detection with Python Tutorial

#### --Anomaly Detection 설명 ([ppt slide]([https://github.com/jy-jeong93/Business-Analytics-IME654/blob/main/2.Kernel-based%20Learning/SVM_slide.pdf](https://github.com/jy-jeong93/Business-Analytics-IME654/blob/main/3.%20Anomaly%20Detection/%EB%B9%84%EC%A6%88%EB%8B%88%EC%8A%A4%EC%95%A0%EB%84%90%EB%A6%AC%ED%8B%B1%EC%8A%A4_%EC%9D%B4%EC%83%81%EC%B9%98%ED%83%90%EC%A7%80.pdf)))--

이번 튜토리얼에서는 anomaly detection을 소개하고 정형 데이터셋을 통해서 다양한 이상치 탐지 기법을 적용해보는 것을 목표로 한다.

# Anomaly detection 방법론
Anomaly detection은 다음과 같이 크게 세 가지 갈래로 나누어 생각할 수 있다.

1. Density/Distance-based methods
  * Gaussian Mixture Model
  * K-Nearest Neighbors(KNN) method
  * LOF(Local Outlier Factors): 데이터 밀도 또는 거리 척도를 통해, majority 군집과 minority 군집을 생성하여 이상치를 탐지

2. Model-based methods
  * Isolation Forest: Tree based method로써 데이터를 분할 및 고립시켜 이상치를 탐지
  * 1-cass SVM: 데이터가 존재하는 영역을 정의하여, 영역 밖의 데이터들은 이상치로 간주

3. Reconstruction-based methods
  * PCA(Principal Component Analysis) method
  * Auto-Encoder based method: 고차원 데이터에서 주로 사용하는 방법론으로써 데이터를 압축/복원하여 복원된 정도로 이상치를 판단


그렇다면 이상치란 무엇이며, anomaly와 novelty 차이는 무엇일까?
비정상 sample의 정의하는 방식에 따른 분류 차이이며 anomaly를 정의하는 방식을 잘 살펴보고 접근해야 한다.

![image](https://user-images.githubusercontent.com/115562646/202421720-309a11b5-dcb4-4be6-839d-46a895a3f5a2.png)
![image](https://user-images.githubusercontent.com/115562646/202422594-a5ab82c1-0073-40ac-a3f7-307c04d230f4.png)




학습시 비정상 sample의 사용 여부 및 label 유무에 따른 분류
![image](https://user-images.githubusercontent.com/115562646/202421786-fee8b044-9838-4524-8272-6fa6ed53a888.png)

비정상 sample 정의에 따른 분류
![image](https://user-images.githubusercontent.com/115562646/202421804-9384d645-4cc9-447e-a2a2-972763d96432.png)





## Isolation Forest, SVM, LOF

Isolation Forest와 SVM, LOF를 사용하여 'creditcard.csv'데이터에 대해 금융 사기 건수 이상치 탐지를 진행할 것이다.
데이터는 총 284,807 건의 거래 데이터이며 각 column 정보는 아래와 같다.

   * V1 ~ V28 : 개인정보로 공개되지 않은 값
   * Time : 시간
   * Amount : 거래금액
   * Class : 사기 여부 (1: fraud 사기, 0: normal 정상) 

|   |  Time |        V1|        V2|...|        V21|        V22|   Amount|  Class|
|:-:|:-----:|:--------:|:--------:|:-:|:---------:|:---------:|:-------:|:-----:|
| 0 |  0.0  | -1.359807| -0.072781|...|   0.133558|  -0.021053|  149.62 |   0   |
| 1 |  0.0  |  1.191857|  0.266151|...|  -0.008983|   0.014724|    2.69 |   0   |
| 2 |  1.0  | -1.358354| -1.340163|...|  -0.055353|  -0.059752|  378.66 |   0   |
| 3 |  1.0  | -0.966272| -0.185226|...|   0.062723|   0.061458|  123.50 |   0   |
| 4 |  2.0  | -1.158233|  0.877737|...|   0.219422|   0.215153|   69.99 |   0   |



총 284,807 건의 거래 데이터 중 492건만이 사기 거래 데이터이며, 이를 이상치로 분류하여 이상치 탐지를 진행할 것이다.

|   |  Target |        Count|        Percent(%)|
|:-:|:-------:|:-----------:|:----------------:|
| 0 |    0    |       284315|             99.83|
| 1 |    1    |          492|              0.17|

또한, 학습용 데이터와 검증용 데이터를 9:1 비율로 나누어 실험을 진행할 것이다.



각 방법론들에 대한 hyperparameter는 다음과 같다.

```python
classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction,random_state=state, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1)
   
}
n_outliers = len(Fraud)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:    
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != Y).sum()
    print("{}: {}".format(clf_name,n_errors))
    print("Accuracy Score :")
    print(accuracy_score(Y,y_pred))
    print("Classification Report :")
    print(classification_report(Y,y_pred))
```

다음은 해당 검증 데이터셋에 대한 각각의 성능을 보여주는 표이다.
|          |1-class SVM|   I-Forest|       LOF|
|:--------:|:---------:|:---------:|:--------:|
| Accuracy |   0.7010  |   0.9974  |  0.9966  |







Anomaly Detection with Python Tutorial

--Anomaly Detection 설명 (ppt slide)--
이번 튜토리얼에서는 anomaly detection을 소개하고 정형 데이터셋과 비정형 데이터셋을 통해서 다양한 이상치 탐지 기법을 적용해보는 것을 목표로 한다.

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
비정상 sample의 정의하는 방식에 따른 분류 차이이며 anomaly를 정의하는 방식을 잘 살펴보고 접근해야 한다
![image](https://user-images.githubusercontent.com/115562646/202421720-309a11b5-dcb4-4be6-839d-46a895a3f5a2.png)

학습시 비정상 sample의 사용 여부 및 label 유무에 따른 분류

![image](https://user-images.githubusercontent.com/115562646/202421786-fee8b044-9838-4524-8272-6fa6ed53a888.png)

비정상 sample 정의에 따른 분류

![image](https://user-images.githubusercontent.com/115562646/202421804-9384d645-4cc9-447e-a2a2-972763d96432.png)



위 방법론들 중에서 Model-based 방법론인 Isolation Forest와 Reconstruction-based 방법론인 Auto-Encoder based method중에서 Variational auto encoder를 사용할 것이다.



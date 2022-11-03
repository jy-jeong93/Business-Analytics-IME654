## Support Vector Machines with Python Tutorial

#### --SVM 설명 ([ppt slide](https://github.com/jy-jeong93/Business-Analytics-IME654/blob/main/2.Kernel-based%20Learning/SVM_slide.pdf))--

이번 튜토리얼에서는 SVM을 소개하고 정형 데이터셋을 통해서 다양한 kernel과 SVM을 적용해보는 것을 목표로 한다.

### 튜토리얼 목차
 1.
 2.
 3.
 4.




## 1. Support Vector Machines(SVM)이란?

Support vector machines(SVM)은 머신러닝 분야 중 하나로써 신호 처리, 의료 응용 분야, 자연어 처리, 음성 및 영상 인식 등 다양한 분야에서 여러 분류 및 회귀 문제에 사용되는 지도 학습 알고리즘이다. 이 알고리즘은 하나의 클래스에 대한 데이터 포인트들을 다른 클래스들의 데이터 포인트들과 최대한 잘 구분해내는 초평면(hyperplane)을 찾는 것을 목표로 한다.
아래 그림을 보면 데이터 포인트들을 나눌 수 있는 분류 경계선은 $2^n$이다.
<p align="center">
 
![image](https://user-images.githubusercontent.com/115562646/199581342-49bcf5c4-d833-49f2-bd87-4483e5d64ea7.png)
 
</p>


## 2. Linear SVMs

Linear SVM은 데이터 포인트들을 최대한 잘 구분해내는 선형분리를 찾는 것이 목적이며, 아래 그림과 같이 두 데이터의 클래스를 분리할 수 있는 수 많은 직선들 중 두 데이터 클래스간 간격(margin)이 최대가 되는 MMH(Maximum Marginal Hyperplane, 최대 마진 초평면)을 찾아 구분하는 방법이다.
![image](https://user-images.githubusercontent.com/115562646/199652997-789ca4a9-59c0-4a2c-ba9f-d587d687d217.png)
![image](https://user-images.githubusercontent.com/115562646/199655813-86c6ea08-e208-4033-9352-8013e36d60c4.png)


## 3. 


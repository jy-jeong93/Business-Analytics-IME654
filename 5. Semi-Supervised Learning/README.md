## Semi-supervised learning - FixMatch 튜토리얼
이번 튜토리얼에서는 semi-supervised learning 방법론 중 FixMatch를 사용하여 WM811k 웨이퍼 빈 맵 데이터셋에 적용한다.


### WM811k 웨이퍼 빈 맵 데이터셋 소개([출처](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map))

해당 데이터셋은 실제 반도체 웨이퍼 빈 맵 불량 패턴을 하기 위한 데이터셋으로써 총 811,457개의 이미지 데이터셋이다.
또한, 다량의 unlabeled 데이터(78.7%, 638,507개)와 소량의 labeled 데이터(21.3%, 172,950개)로 구성되어있다.

![image](https://user-images.githubusercontent.com/115562646/209644138-da739f69-9615-4ab5-b92c-7b6ef33dc961.png)


단순히 labeled 데이터만 사용하는 supervised learning보다 semi-supervised learning을 통해 분류 성능 개선을 기대할 수 있을 것이며, 동시에 데이터 레이블링 과정에서 발생하는 시간과 비용 문제를 개선할 수 있을 것이다.

![image](https://user-images.githubusercontent.com/115562646/209645858-b2ac4a38-af75-4c97-9a11-b8d11ab8823d.png)


#### (1) Augmentation 정의
Fixmatch 방법론에서는 이미지에 대한 weak augmentation, strong augmentation을 정의해야 한다. 하지만 wm811k데이터셋은 RGB 3차원의 컬러 이미지가 아닌 1차원의 gray scale 이미지이다. 따라서 해당 데이터셋 특성에 맞게 augmentation을 임의로 정의하였다.
아래 그림은 augmentation에 대한 예시이며, strong augmentation에 cutout을 기본적으로 사용한 이유는 FixMatch 본 논문에서 사용했던 strong augmentation 기법 중 유일하게 적용이 가능한 기법이기 때문이다.

![image](https://user-images.githubusercontent.com/115562646/209646589-625bd8df-4603-4f3f-b241-4a50f0b27b28.png)


#### (2) 평가 지표 정의(Macro F1-score)
해당 데이터셋은 공정 데이터셋이므로 정상 패턴이 불량 패턴에 비해 압도적으로 많다. 따라서 단순 accuracy를 사용하는 것이 아닌 Macro F1-score를 평가 지표로써 사용하였다.

![image](https://user-images.githubusercontent.com/115562646/209646976-3b5e0687-fc48-4797-b791-304cf26d9203.png)
![image](https://user-images.githubusercontent.com/115562646/209647004-7fe6b567-cfd4-44e3-a399-a91e42ed174f.png)


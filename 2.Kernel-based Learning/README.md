# 2022 Business Analytics Topic 1 Tutorial
__2022010558 김지현__  
1. __Supervised Method__ 중 변수 선택 기법 중 하나인 [Genetic Algorithm(유전 알고리즘)에 대한 튜토리얼](https://github.com/Im-JihyunKim/BusinessAnalytics_Topic1/blob/main/Supervised%20Dimensionality%20Reduction/GA_Feature_Selection_Tutorial.ipynb)을 작성하였습니다.   
2. __Unsupervised Method__ 중 Linear Embedding 기법 중 [MDS(다차원 척도법)에 대한 튜토리얼](https://github.com/Im-JihyunKim/BusinessAnalytics_Topic1/blob/main/Unsupervised%20Dimensionality%20Reduction/MDS_Tutorial.ipynb)을 작성하였습니다.  
링크를 클릭하면 보다 상세한 튜토리얼을 확인할 수 있습니다.

## Table of Contents:
- [Dimensionality Reduction](#dimensionality-reduction)
  * [Supervised Dimensionality Redcution: Genetic Algorithm](#supervised-dimensionality-redcution--genetic-algorithm)
    + [How Genetic Algorithm Works](#how-genetic-algorithm-works)
    + [Genetic Algorithm Ending Criteria](#genetic-algorithm-ending-criteria)
    + [Fitness Evaluation](#fitness-evaluation)
    + [Selection](#selection)
    + [Crossover and Mutation](#crossover-and-mutation)
    + [Requirements](#requirements)
    + [Parameters](#parameters)
    + [Argparse](#argparse)
    + [Example of Use](#example-of-use)
  * [Multidimensional Reduction (MDS)](#multidimensional-reduction--mds-)
    + [Purpose](#purpose)
    + [How to Use](#how-to-use)
    + [Parameters](#parameters-1)
    + [Simple Illustration](#simple-illustration)
- [References](#references)
    + [Genetic Algorithm](#genetic-algorithm)
    + [Multidimensional Scaling](#multidimensional-scaling)

# Dimensionality Reduction
이미지, 텍스트, 센서 등 다양한 도메인의 데이터들은 변수의 수가 매우 많은 고차원 데이터(High Dimensional Data)의 특징을 가지고 있습니다. 그러나 많은 기계학습 알고리즘은 실제 데이터 차원을 모두 사용하지 않고, 정보를 축약하여 내재된 차원(Intrinsic/Embedded Dimension)을 활용하는 경우가 많습니다. 이는 __차원의 저주(curse of Dimensionality)__ 를 해결하기 위함인데, 사용하는 변수 수를 줄이면 잡음(noise)이 포함될 확률도 감소시킴과 동시에 예측 모델의 성능을 높이고, 예측 모델의 학습과 인식 속도를 빠르게 할 수 있으며 예측 모델에 필요한 학습 집합의 크기를 크게 할 수 있기 때문입니다.   
따라서 분석 과정에서 성능을 저하시키지 않는 최소한의 변수 집합을 판별하여 주요 정보만을 보존하는 것이 중요하며, 차원 축소 방식은 __(1) Supervised Dimensionality Reduction (교사적 차원 축소)__ , __(2) Unupservised Deimensionality Reduction (비교사적 차원 축소)__ 두 가지로 구분할 수 있습니다. Supervised Dimensionality Reduction은 축소된 차원의 적합성을 검증하는 데 있어 예측 모델을 적용하며, 동일한 데이터라도 적용되는 모델에 따라 임베딩 결과가 달라질 수 있다는 특징을 가집니다. 반면 Unupservised Deimensionality Reduction은 축소된 차원의 적합성을 검증하는 데 있어 예측 모델을 적용하지 않고, 특정 기법에 따라서 차원 축소 결과는 언제나 동일하다는 특징을 가집니다.  
본 튜토리얼에서는 __Supervised Dimensionality Reduction__ 방법론 중 활용하는 변수의 수를 줄이는 __Feature Selection (변수 선택)__ 방법론 중 __Genetic Algorithm (유전 알고리즘)에 초점을 맞추어 차원 축소를 수행__ 해보고자 합니다. 또한 __Unupservised Deimensionality Reduction__ 방법론 중에서는 Linear Embedding 방법론 중 __Multidimensional Scaling (다차원 척도법)__ 에 초점을 맞추어 튜토리얼을 작성하였습니다.

## Supervised Dimensionality Redcution: Genetic Algorithm
유전 알고리즘은 변수 선택 기법 중 가장 우수한 방법입니다. 이전까지의 변수 선택 기법들은 탐색 소요 시간을 줄여 효율적인 방법론을 제안하였으나, 탐색 범위가 적어 Global Optimum을 찾을 확률이 적은 한계를 가지고 있었습니다. 그러나 __자연계의 진화 체계를 모방한 메타 휴리스틱 알고리즘__ 인 GA는 시행착오를 통해 최적의 해를 찾아나가는 방법론으로, 다윈의 자연 선택설에 기반하여 초기에 다양한 유전자를 가지고 있던 종이 생존에 유리한 유전자를 택하면서 현재 상태가 되었다는 이론을 따라 해를 최적화 해나갑니다.
> **Heuristic 휴리스틱**   
> 참고로 휴리스틱이란 불충분한 시간이나 정보로 인하여 합리적인 판단을 할 수 없거나, 체계적이면서 합리적인 판단이 굳이 필요하지 않은 상황에서 사람들이 빠르게 사용할 수 있게 보다 용이하게 구성된 간편추론 방법론을 의미합니다. **메타 휴리스틱(Meta Heuristic)** 은 휴리스틱 방법론 중 풀이 과정 등이 구조적으로 잘 정의되어 있어 대부분의 문제에 어려움 없이 적용할 수 있는 휴리스틱을 의미합니다.

### How Genetic Algorithm Works
유전 알고리즘은 기본적으로 여러 개의 해로 구성된 잠재 해 집단을 만들고 적합도(fitness)를 평가한 뒤, 좋은 해를 선별해서 새로운 해 집단(후기 세대)을 만드는 메타 휴리스틱 알고리즘입니다. 진화 이론 중 자연선택설에 기반하여 세대를 생성해내며, 주어진 문제를 잘 풀기 위한 최적해를 찾거나 종료 조건을 만족 시 알고리즘이 종료됩니다. 후기 세대를 만드는 과정은 (부모 세대) __선택(Selection)__ , __교배(Corssover)__ , __돌연변이 발생(Mutation)__ 3가지에 기반하며, 한 세대(잠재해 집단)는 __적합도 함수(Fitness function)__ 에 의해 문제 해결에 적합한지 평가됩니다. 기본적으로 Genetic Algorithm은 최적화 문제에서 사용되지만, 본 튜토리얼에서는 차원 축소 시 목표변수를 예측하는 데 사용되는 설명 변수 조합을 선택하는 데 Genetic Algorithm을 사용하였습니다. 본 알고리즘을 도식화하면 아래와 같습니다.
![image](https://user-images.githubusercontent.com/115214552/195269445-768a0a06-c8ad-43a4-9a1a-a4d0c5d331fb.png)

__Genetic Algorithm Process:__
1. 초기 세대 생성
2. 세대 적합도 평가([Fitness Evaluation](#fitness-evaluation))
3. 부모 세대 선택([Selection](#selection))
4. 교차 및 돌연변이 생성을 통한 자식 세대 생성([Crossover & Mutation](#crossover-and-mutation))
5. 자식 세대 적합도 평가

### Genetic Algorithm Ending Criteria
유전 알고리즘은 위 5단계 Process를 거치며 알고리즘 종료 조건을 만족한 경우 학습이 완료됩니다.
1. 사용자가 지정한 세대 수(`n_generation`)를 모두 생성한 경우
2. 학습 시 모델이 수렴한 경우
 - 이는 `threshold_times_convergence` 횟수를 넘어가는 동안 최고 성능을 갱신하지 못하는 경우에 해당합니다. 즉, 알고리즘이 local optimal을 찾아 종료된 경우입니다. `threshold_times_convergence`는 초기에 5번으로 상정하였습니다.
 -  만일 `n_genetation`의 절반 이상 학습이 진행되었다면 조금 더 증가하여 global optimal을 찾도록 하였습니다. `threshold_times_convergence`를 생성된 세대 수의 30% 만큼으로 지정하여, 해당 수 이상으로 Score 값이 일정하다면 학습을 종료합니다.
 -  더불어 새로운 자식 세대의 최고 성능과 전 세대의 최고 성능 간 차이가 지정한 `threshold` 보다 낮다면, `threshold_times_convergence` 횟수만큼 반복될 경우 학습을 조기 종료합니다.

### Fitness Evaluation
- 적합도 평가는 각 염색체(Chromosome)의 정보를 사용하여 학습된 모형의 적합도를 평가하는데, 염색체의 우열을 가릴 수 있는 정략적 지표를 통해서 높은 값을 가질 수록 우수한 염색체(변수 조합)으로서 채택합니다.
- 적합도 함수(Fitness Function)가 가져야 하는 두 가지 조건은 다음과 같습니다.
    1. 두 염색체가 __동일한 예측 성능__ 을 나타낼 경우, __적은 수의 변수__ 를 사용한 염색체 선호
    2. 두 염색체가 __동일한 변수__ 를 사용했을 경우, __우수한 예측 성능__ 을 나타내는 염색체 선호
- 본 튜토리얼에서 Classification Task를 위해 사용한 모델은 Logistic Regression이며, 적합도 평가를 위해 사용한 척도는 __(1) Accuracy__ , __(2) F1-Score__ , __(3) AUROC Score__ 3가지가 있습니다. Regression Task를 위해 사용한 모델은 Linear Regression 혹은 다른 어떤 모델이든 상관 없으며, 적합도 평가를 위해 사용한 척도는  __(1) 스피어만 상관계수, (2) MAPE, (2) RMSE, (4) MAE를 1에서 빼준 값__ 으로 사용합니다.

### Selection
- 적합도 함수를 통해서 부모 염색체의 우수성을 평가하였다면, Step 3에서는 우수한 부모 염색체를 선택하여 자손에게 물려줍니다. 이는 부모 염색체가 우월하다면, 자손들도 우월할 것이라는 가정에 기반합니다. 이때 부모 염색체를 선택하는 방법은 여러 가지이고, 대표적인 방법론들은 아래와 같습니다.
    1. __Deterministic Selection__  
       - 적합도 평가 결과로 산출된 rank 기준으로 상위 N%의 염색체를 선택하는 것입니다. 우수한 유전자를 물려주어 좋은 해를 만들어내기 위한 방법론입니다. 그러나 상위 N%보다 아래의 염색체 중 적합도에 차이가 얼마 나지 않는 경우를 반영하지 못한다는 한계가 존재합니다. 이를 보완한 방법이 Probabilistic Selection입니다.
    2. __Probabilistic Selection__
       - 각 염색체에 가중치를 부여하여, 모든 염색체에게 자손에게 전달해 줄 수 있는 기회를 부여하는 방법론입니다. 룰렛 휠 방식(Roulette Wheel Selection)이라고도 하며, Classification Task에서는 Softmax 확률 값에 기반하여 가중치를 부여할 수 있습니다.
    3. __Tournament Selection__
       - 무작위로 K개의 염색체를 선택하고, 이들 중 가장 우수한 염색체를 택하여 다음 세대로 전달하는 방법론입니다. 동일한 프로세스가 다음 상위 염색체를 선택하기 위해 반복되며, Deterministic Selection의 단점을 어느정도 보완한 동시에 연산 시간이 비교적 짧다는 장점을 가집니다.
- 본 튜토리얼에서는 염색체 세대가 언제나 동일해야 한다는 점에 기반하여, __Tournament Selection을 이용하여 선택을 진행__ 하였습니다.  가장 적합도가 높은 염색체를 선정한 이후에, 무작위로 K개의 염색체를 골라 적합도 Score를 비교하고, 높은 염색체를 고르는 과정을 세대 수만큼 반복하여 다음 세대를 만드는 것입니다. 해당 방법론의 개요를 도식화 하면 아래와 같습니다.

<p align="center">
     <img src="https://user-images.githubusercontent.com/115214552/195280272-066c04b5-fe53-4fdb-9ead-6e81edfd5f9b.png" alt="tournament selection"/>
</p>

### Crossover and Mutation
![image](https://user-images.githubusercontent.com/115214552/195280032-02f005bd-48ae-4221-a4cc-98056240fc71.png)

__Crossover 교배__
- 선택된 부모 염색체로부터 자식세대를 재생산해내는 과정입니다. 
- 앞 단계에서 선택된 부모 염색체들의 유전자 정보를 서로 교환하여 새로운 자식 염색체들을 최종적으로 생성해냅니다.
- 본 튜토리얼에서는 교배율을 Hyperparameter로 지정하여, 얼마나 많은 변수들을 교환하여 자식 염색체를 생성해낼 지를 자유롭게 지정할 수 있게 하였습니다.
- 본 튜토리얼에서 사용된 교배율(crossover_rate)은 0.7입니다.

__Mutation 돌연변이__
- 돌연변이는 세대가 진화해 가는 과정에서 다양성을 확보하기 위한 장치입니다.
- 특정 유전자의 정보를 낮은 확률로 반대 값으로 변환하는 과정을 통해 돌연변이를 유도합니다.
- 돌연변이를 통해 현재 해가 Local Optimum에서 탈출할 수 있는 기회를 제공하지만, 너무 높은 돌연변이율은 유전 알고리즘의 convergence 속도를 늦추기에 주로 0.01 이하의 값을 사용합니다.


### Requirements
- Python >= 3.6
- numpy >= 1.18
- pandas >= 1.0.1
- rich >= 12.6.0
- scikit-learn >= 1.1.2

### Parameters
Genetic Algorithm class를 호출하는 데 필요한 파라미터 목록입니다.
|__Parameter__|__Type__|__Default__|__Definition__|
|------|---|---|---|
|`model`|object||Scikit-learn에서 제공하는 기본 지도학습 머신러닝 알고리즘이어야 합니다. fit, predict 등의 method를 지원해야 합니다.|
|`args`|argparse||유전 알고리즘에 필요한 여러 하이퍼파라미터를 정의할 수 있습니다.|
|`seed`|int|2022|각 세대를 만들어냄에 있어 Randomness를 제어하기 위함입니다. 정수값을 입력합니다.|

### Argparse
유전 알고리즘에서 필요한 하이퍼파라미터 목록입니다. 터미널에서 `main.py`를 실행 시 인자 값을 자유롭게 바꿀 수 있습니다.
|__Argument__|__Type__|__Default__|__Help__|
|------|---|---|---|
|`seed`|int|2022|각 세대를 만들어냄에 있어 Randomness를 제어하기 위함입니다. 정수값을 입력합니다.|
|`normalization`|bool|False|입력 데이터 값 Scaling 여부입니다.|
|`n_generation`|int|50|얼마나 많은 세대를 만들어낼 지를 결정하는 부분으로, 알고리즘 종료조건 중 하나입니다.|
|`n_population`|int|100|한 세대에 얼마나 많은 염색체 수(변수 조합)를 고려할 것인지를 결정합니다. 값이 클 수록 연산량이 많아지지만 더 많은 범위를 탐색할 수 있습니다.|
|`crossover_rate`|float|0.7|유전자 정보를 얼마나 교환하여 자식 세대를 생성할 지 비율을 지정합니다. 0.0에서 1.0 사이의 값을 가져야 합니다.|
|`mutation_rate`|float|0.1|자식 세대에서 돌연변이를 얼마나 만들어낼 지를 비율을 지정합니다. 0.0에서 1.0 사이의 값을 가져야 합니다.|
|`tournament_k`|int|2|본 튜토리얼은 Selection 시 Tournament Selection 방식을 택했습니다. 부모 세대로 선택하기 위한 과정 중 K개의 염색체를 무작위로 골라 토너먼트를 진행합니다.|
|`c_metric`|str|'accuracy'|Classification Task에서의 적합도 평가를 위한 지표입니다. accuracy, f1-score, roc_auc_score 3가지를 선택하여 사용할 수 있습니다.|
|`r_metric`|str|'rmse'|Regression Task에서의 적합도 평가를 위한 지표입니다. corr, rmse, mape, mae 4가지 중 하나를 선택하여 사용할 수 있습니다.|
|`n_jobs`|int|1|CPU 코어를 얼마나 사용할 지를 정하는 인자입니다. -1로 지정 시 컴퓨터의 모든 코어를 사용하게 됩니다.|
|`initial_best_chromosome`|np.ndarray|None|1차원의 이진화된 매트릭스로, 데이터의 변수 개수 만큼의 크기를 갖습니다. 초기 세대에서의 최고 염색체가 무엇인지를 결정하는 인자입니다.|
|`verbose`|int|0|함수 수행 시 출력되는 정보들을 얼마나 상세히 할 지를 결정하는 인자입니다. 0은 출력하지 않고, 1은 자세히, 2는 함축적 정보만 출력합니다.|

### Example of Use
```python
import argparse
import numpy as np
import rich
import argparse
from ga_feature_selection.genetic_algorithm import GA_FeatureSelector
from sklearn import datasets
from sklearn.datasets import make_classification, make_regression
from sklearn import linear_model


def main(args):
    """Loading X(features), y(targets) from datasets"""
    data = datasets.load_wine()
    X, y = data['data'], data['targets']
    LogisticRegression = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    Genetic_Algorithm = GA_FeatureSelector(model=LogisticRegression, args=args, seed=args.seed)
    
    """Making train and test set"""
    X_train, X_test, y_train, y_test = Genetic_Algorithm.data_prepare(X, y)
    Genetic_Algorithm.run(X_train, X_test, y_train, y_test)

    """Show the result"""
    table, summary_table = Genetic_Algorithm.summary_table()
    rich.print(table)
    rich.print(summary_table)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--normalization", default=False, type=bool)
    parser.add_argument("--n-generation", default=10, type=int, help="Determines the maximum number of generations to be carry out.")
    parser.add_argument("--n-population", default=100, type=int, help="Determines the size of the population (number of chromosomes).")
    parser.add_argument("--crossover-rate", default=0.7, type=float, help="Defines the crossing probability. It must be a value between 0.0 and 1.0.")
    parser.add_argument("--mutation-rate", default=0.1, type=float, help="Defines the mutation probability. It must be a value between 0.0 and 1.0.")
    parser.add_argument("--tournament-k", default=2, type=int, help="Defines the size of the tournament carried out in the selection process. \n 
                         Number of chromosomes facing each other in each tournament.")
    parser.add_argument("--n-jobs", default=1, choices=[1, -1], type=int, help="Number of cores to run in parallel. By default a single-core is used.")
    parser.add_argument("--initial-best-chromosome", default=None, type=np.ndarray, 
                        help="A one-dimensional binary matrix of size equal to the number of features (M). \n
                        Defines the best chromosome (subset of features) in the initial population.")
    parser.add_argument("--verbose", default=0, type=int, help="Control the output verbosity level. It must be an integer value between 0 and 2.")
    parser.add_argument("--c-metric", default='accuracy', choices=['accuracy', 'f1_score', 'roc_auc_socre'], type=str)
    parser.add_argument("--r-metric", default='rmse', choices=['rmse', 'corr', 'mape', 'mae'], type=str)
    
    args = parser.parse_args()
    
    main(args)
```
```
Creating initial population with 100 chromosomes 🧬
 ✔ Evaluating initial population...
 ✔ Current best chromosome: [1 0 0 0 0 1 1 0 0 1 0 1 1], Score: 0.971830985915493
Creating generation 1...
 ✔ Evaluating population of new generation 1...
 ✔ (Better) A better chromosome than the current one has been found 0.9859154929577465
 ✔ Current best chromosome: [1 1 1 1 0 1 1 1 1 1 0 1 0], Score: 0.9859154929577465
    Elapsed generation time:  2.73 seconds
Creating generation 2...
 ✔ Evaluating population of new generation 2...
 ✔ Same scoring value found 1 / 5 times.
 ✔ Current best chromosome: [1 1 1 1 0 1 1 1 1 1 0 1 0], Score: 0.9859154929577465
    Elapsed generation time:  2.71 seconds
Creating generation 3...
 ✔ Evaluating population of new generation 3...
 ✔ Same scoring value found 2 / 5 times.
 ✔ Current best chromosome: [1 1 1 1 0 1 1 1 1 1 0 1 0], Score: 0.9859154929577465
    Elapsed generation time:  2.69 seconds
(...)
Creating generation 49...
 ✔ Evaluating population of new generation 49...
 ✔ (WORSE) No better chromosome than the current one has been found 0.971830985915493
 ✔ Current best chromosome: [1 0 1 1 0 0 1 0 0 0 1 0 0], Score: 0.9929577464788732
    Elapsed generation time:  2.76 seconds
Creating generation 50...
 ✔ Evaluating population of new generation 50...
 ✔ (WORSE) No better chromosome than the current one has been found 0.9788732394366197
 ✔ Current best chromosome: [1 0 1 1 0 0 1 0 0 0 1 0 0], Score: 0.9929577464788732
    Elapsed generation time:  2.71 seconds
Training time:  138.77 seconds
```

결과(table, summary table)는 아래와 같습니다.

```
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃                  ┃     Selected     ┃                  ┃                 ┃                  ┃   Training Time   ┃
┃ Best Chromosome  ┃   Features ID    ┃ Best Test Score  ┃ Best Generation ┃ Best Train Score ┃       (sec)       ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ [1 0 1 1 0 0 1 0 │ [ 0  2  3  6 10] │ 0.9929577464788… │        4        │       1.0        │      138.77       │
│    0 0 1 0 0]    │                  │                  │                 │                  │                   │
└──────────────────┴──────────────────┴──────────────────┴─────────────────┴──────────────────┴───────────────────┘
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Number of Generation ┃ Number of Population ┃ Crossover Rate ┃ Mutation Rate ┃  Metric  ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│          50          │         100          │      0.7       │      0.1      │ accuracy │
└──────────────────────┴──────────────────────┴────────────────┴───────────────┴──────────┘
```

## Multidimensional Reduction (MDS)
MDS는 데이터를 저차원 공간으로 mapping 함에 있어 non-linear 방식을 사용하는 기법입니다. 고차원 공간에 있는 점을 저차원 공간에 mapping하면서, 해당 점 사이의 거리를 최대한 유지하는 것이 MDS의 목적입니다. 즉, 저차원 공간 상에서 데이터의 Pairwise Distance는 고차원 공간의 실제 거리와 거의 일치해야 합니다. MDS는 Classification 및 Regression Task에서도 전처리 단계 차원에서 사용할 수 있는데, 변수를 축소할뿐 아니라 데이터를 시각화함에 있어서도 효과적인 기술입니다. 저차원 공간에서도 고차원의 원본 데이터와 동일한 Cluster와 Pattern을 유지하기 때문에, 일례로 5차원의 데이터가 있다고 하더라도 3차원 데이터로 만들어 시각화할 수 있는 것입니다.  
일반적으로 MDS에서 데이터 간 Pairwise Distance를 구하는 방법은 유클리디안 거리를 이용하지만, 다른 적절한 metric을 이용하여 비교할 수도 있습니다. 본 튜토리얼에서는 Scikit-learn 라이브러리를 사용하여서, python을 이용해 다차원 척도법을 구현해보고자 합니다. 간단한 예제를 통해서 MDS 적용 방법론을 알아보겠습니다.

### Purpose
MDS는 D차원의 공간 상에 객체들(데이터)이 있다고 했을 때, 해당 객체들의 거리가 저차원 공간 상에서도 최대한 많이 보존되도록 하는 축, 좌표계를 찾는 것입니다.  
아래 예시를 보겠습니다. 만일 미국의 두 도시들 간의 비행 거리를 통해서 각 도시들이 얼마나 떨어져 있는지를 계산한 Distance Matrix가 주어져 있다 가정하겠습니다. 해당 예에 따르면 보스턴과 뉴욕은 206, 보스턴과 DC는 409, 보스턴과 마이애미는 1504 만큼 거리 차이가 있으며, 이처럼 두 객체들 간에 Pairwise Distance를 2차원 공간 상에 각 도시들을 Mapping하면 2차원 축, 좌표로 표현될 수 잇을 것입니다. 결국 MDS는 이러한 Distance/Similarity Matrix를 통해서 저차원 상의 각 객체들이 갖는 좌표(Coordinates) 시스템을 찾는 것을 목적으로 합니다. 이는 주성분 분석(PCA)이 데이터 특징을 데이터가 가지는 분산으로 정의한 것과는 분명히 다른 지점입니다.
![image](https://user-images.githubusercontent.com/115214552/195580072-a3a73167-9dd7-4f27-8cba-d2e3ed8e0132.png)

### How to Use
Scikit-Learn 라이브러리의 `sklearn.manifold` 모듈에서는 다양한 학습 데이터에 대한 embedding 기술을 구현하고 있습니다. 이때 해당 모듈에서 제공하는 'MDS' 클래스를 사용하여 다차원 척도법을 구현할 수 있습니다.
### Parameters
MDS Class 사용을 위한 파라미터 목록은 아래와 같습니다.
|__Parameter__|__Type__|__Default__|__Definition__|
|------|---|---|---|
|`n_components`|int|2|데이터를 몇 차원으로 줄일 지를 결정하는 인자입니다. 기본 값은 2입니다.|
|`metric`|bool|True|Metric MDS의 경우 True, Non-metric MDS의 경우 False를 입력합니다.|
|`dissimilarity`|str|'euclidean'|객체들 간의 거리, 유사성, 비유사성을 구하는 척도를 입력합니다. 혹은 'precomputed'를 통해 미리 계산된 Distance Matrix를 입력 값으로 활용할 수도 있습니다.|

### Simple Illustration
아래처럼 임의의 3차원 데이터셋이 있다고 하였을 때, 이를 저차원(2차원)으로 축소시킨 결과를 확인해보겠습니다.
- Scikit-learn에서 제공하는 방식을 통해 데이터를 축소하고 그 결과를 시각화 한 것입니다.
- 2차원 공간 상에서 mapping된 좌표를 보면, 원본 차원에서의 데이터 포인트 상의 거리를 거의 동일하게 유지하는 것을 알 수 있습니다. 오렌지, 갈색, 분홍색의 포인트들이 매우 가깝게 몰려있는 것이 그 예입니다.
```Python
# Make dataset
X = np.random.uniform(1, 10, size=(10, 3))
rich.print('[black]Raw X data: ', '\n', f'[black]{X}')
```
```
Raw X data:  
 [[8.40240565 8.27954781 8.37334735]
 [7.23594263 1.27198244 3.75899031]
 [8.40197083 1.25961067 9.92239   ]
 [2.604193   7.08485018 9.5470354 ]
 [7.74940562 8.23301604 8.10162561]
 [3.58455736 9.95850387 8.86042996]
 [4.93951747 7.32528616 5.80387287]
 [6.41656317 3.58980413 8.47360454]
 [6.84648273 2.2235792  6.88919865]
 [7.4398821  8.90426094 7.02083169]]
```
```Python
# MDS Results using Euclidean Distance
mds = MDS(dissimilarity='precomputed', random_state=2022)
X_transform = mds.fit_transform(X)
rich.print('[black]MDS Coordinates: ', '\n', f'[black]{X_transform}')
```
```
MDS Coordinates:  
 [[-1.61079884 -2.95430259]
 [ 0.04169207  6.10767705]
 [-4.55096644  3.07620942]
 [ 4.40971489 -0.87276813]
 [-0.96938912 -2.6982923 ]
 [ 3.51228508 -3.87945677]
 [ 1.76924661 -0.74915278]
 [-1.16040954  1.78236135]
 [-1.19244475  3.49704443]
 [-0.24892997 -3.30931968]]
```

```Python
# Result Visualization
colors = ['darkorange', 'midnightblue', 'salmon', 'saddlebrown', 'peru',
          'darkcyan', 'indigo', 'darkseagreen', 'mediumseagreen', 'pink']
size = [64] * X.shape[0]
fig = plt.figure(2, (20, 10))
ax = fig.add_subplot(121, projection='3d')
plt.scatter(X[:,0], X[:,1], zs=X[:,2], s=size, c=colors)
plt.title('Original Points')

ax = fig.add_subplot(122)
plt.scatter(X_transform[:,0], X_transform[:,1], s=size, c=colors)
plt.title('Embedding in 2D')
fig.subplots_adjust(wspace=.4, hspace=0.5)
plt.show()
```
![output](https://user-images.githubusercontent.com/115214552/195580989-410e1870-0624-4cad-867c-fc697613536e.png)

# References
### Genetic Algorithm
- R. Tolosana, J.C. Ruiz-Garcia, R. Vera-Rodriguez, J. Herreros-Rodriguez, S. Romero-Tapiador, A. Morales and J. Fierrez, "Child-Computer Interaction: Recent Works, New Dataset, and Age Detection", IEEE Transactions on Emerging Topics in Computing, doi: 10.1109/TETC.2022.3150836, 2022.
- https://featureselectionga.readthedocs.io/en/latest/

### Multidimensional Scaling
- https://stackabuse.com/guide-to-multidimensional-scaling-in-python-with-scikit-learn/
- https://github.com/klyshko/ml_python/blob/master/Lecture9.ipynb

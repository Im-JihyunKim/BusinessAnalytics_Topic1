# 2022 Business Analytics Topic 1 Tutorial
__2022010558 김지현__  
1강 차원 축소 Topic에서, 변수 선택 방법론 중 Genetic Algorithm(유전 알고리즘)에 대한 튜토리얼을 작성하였습니다.
## Table of Contents:
- [Dimensionality Reduction](#dimensionality-reduction)
  * [Genetic Algorithm](#genetic-algorithm)
    + [How Genetic Algorithm Works](#how-genetic-algorithm-works)
    + [Genetic Algorithm Ending Criteria](#genetic-algorithm-ending-criteria)
    + [Fitness Evaluation](#fitness-evaluation)
    + [Selection](#selection)
    + [Crossover and Mutation](#crossover-and-mutation)
  * [Requirements](#requirements)
  * [Parameters](#parameters)
  * [Argparse](#argparse)
  * [Example of Use](#example-of-use)
- [References](#references)

# Dimensionality Reduction
이미지, 텍스트, 센서 등 다양한 도메인의 데이터들은 변수의 수가 매우 많은 고차원 데이터(High Dimensional Data)의 특징을 가지고 있습니다. 그러나 많은 기계학습 알고리즘은 실제 데이터 차원을 모두 사용하지 않고, 정보를 축약하여 내재된 차원(Intrinsic/Embedded Dimension)을 활용하는 경우가 많습니다. 이는 __차원의 저주(curse of Dimensionality)__ 를 해결하기 위함인데, 사용하는 변수 수를 줄이면 잡음(noise)이 포함될 확률도 감소시킴과 동시에 예측 모델의 성능을 높이고, 예측 모델의 학습과 인식 속도를 빠르게 할 수 있으며 예측 모델에 필요한 학습 집합의 크기를 크게 할 수 있기 때문입니다.   
따라서 분석 과정에서 성능을 저하시키지 않는 최소한의 변수 집합을 판별하여 주요 정보만을 보존하는 것이 중요하며, 대표적인 방법론으로는 활용하는 변수의 수를 줄이는 __Feature Selection (변수 선택)__ , 그리고 새로운 변수를 만들어서 활용 변수 개수를 줄이는 __Feature Extraction (변수 추출)__ 두 가지 방법을 소개할 수 있습니다. 이때 본 튜토리얼에서는 변수 선택 기법 중 __Genetic Algorithm (유전 알고리즘)에 초점을 맞추어 차원 축소를 수행__ 해보고자 합니다.

## Genetic Algorithm
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
- 본 튜토리얼에서는 염색체 세대가 언제나 동일해야 한다는 점에 기반하여, __Tournament Selection을 이용하여 선택을 진행__하였습니다.  가장 적합도가 높은 염색체를 선정한 이후에, 무작위로 K개의 염색체를 골라 적합도 Score를 비교하고, 높은 염색체를 고르는 과정을 세대 수만큼 반복하여 다음 세대를 만드는 것입니다. 해당 방법론의 개요를 도식화 하면 아래와 같습니다.
![image.png](attachment:image.png)
[Tournament Selection]

### Crossover and Mutation
__Crossover 교배__
- 선택된 부모 염색체로부터 자식세대를 재생산해내는 과정입니다. 
- 앞 단계에서 선택된 부모 염색체들의 유전자 정보를 서로 교환하여 새로운 자식 염색체들을 최종적으로 생성해냅니다.
- 본 튜토리얼에서는 교배율을 Hyperparameter로 지정하여, 얼마나 많은 변수들을 교환하여 자식 염색체를 생성해낼 지를 자유롭게 지정할 수 있게 하였습니다.
- 본 튜토리얼에서 사용된 교배율(crossover_rate)은 0.7입니다.

__Mutation 돌연변이__
- 돌연변이는 세대가 진화해 가는 과정에서 다양성을 확보하기 위한 장치입니다.
- 특정 유전자의 정보를 낮은 확률로 반대 값으로 변환하는 과정을 통해 돌연변이를 유도합니다.
- 돌연변이를 통해 현재 해가 Local Optimum에서 탈출할 수 있는 기회를 제공하지만, 너무 높은 돌연변이율은 유전 알고리즘의 convergence 속도를 늦추기에 주로 0.01 이하의 값을 사용합니다.
![image](https://user-images.githubusercontent.com/115214552/195277889-6b6f54e7-a631-4006-8cf0-81c74d9250bb.png)


## Requirements
- Python >= 3.6
- numpy >= 1.18
- pandas >= 1.0.1
- rich >= 12.6.0

## Parameters
Genetic Algorithm class를 호출하는 데 필요한 파라미터 목록입니다.
|__Parameter__|__Type__|__Default__|__Definition__|
|------|---|---|---|
|`model`|object||Scikit-learn에서 제공하는 기본 지도학습 머신러닝 알고리즘이어야 합니다. fit, predict 등의 method를 지원해야 합니다.|
|`args`|argparse||유전 알고리즘에 필요한 여러 하이퍼파라미터를 정의할 수 있습니다.|
|`seed`|int|2022|각 세대를 만들어냄에 있어 Randomness를 제어하기 위함입니다. 정수값을 입력합니다.|

## Argparse
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

## Example of Use
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

# References
- R. Tolosana, J.C. Ruiz-Garcia, R. Vera-Rodriguez, J. Herreros-Rodriguez, S. Romero-Tapiador, A. Morales and J. Fierrez, "Child-Computer Interaction: Recent Works, New Dataset, and Age Detection", IEEE Transactions on Emerging Topics in Computing, doi: 10.1109/TETC.2022.3150836, 2022.
- https://featureselectionga.readthedocs.io/en/latest/

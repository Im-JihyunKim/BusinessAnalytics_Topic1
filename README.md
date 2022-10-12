# 2022 Business Analytics Topic1 Tutorial
### Table of contents:
- [Dimensionality Reduction](#dimensionality-reduction)
- [Genetic Algorithm](#genetic-algorithm)
    + [기본 개념](#-----)

# Dimensionality Reduction
이미지, 텍스트, 센서 등 다양한 도메인의 데이터들은 변수의 수가 매우 많은 고차원 데이터(High Dimensional Data)의 특징을 가지고 있습니다. 그러나 많은 기계학습 알고리즘은 실제 데이터 차원을 모두 사용하지 않고, 정보를 축약하여 내재된 차원(Intrinsic/Embedded Dimension)을 활용하는 경우가 많습니다. 이는 __차원의 저주(curse of Dimensionality)__ 를 해결하기 위함인데, 사용하는 변수 수를 줄이면 잡음(noise)이 포함될 확률도 감소시킴과 동시에 예측 모델의 성능을 높이고, 예측 모델의 학습과 인식 속도를 빠르게 할 수 있으며 예측 모델에 필요한 학습 집합의 크기를 크게 할 수 있기 때문입니다.   
따라서 분석 과정에서 성능을 저하시키지 않는 최소한의 변수 집합을 판별하여 주요 정보만을 보존하는 것이 중요하며, 대표적인 방법론으로는 활용하는 변수의 수를 줄이는 __Feature Selection (변수 선택)__ , 그리고 새로운 변수를 만들어서 활용 변수 개수를 줄이는 __Feature Extraction (변수 추출)__ 두 가지 방법을 소개할 수 있습니다. 이때 본 튜토리얼에서는 변수 선택 기법 중 __Genetic Algorithm (유전 알고리즘)에 초점을 맞추어 차원 축소를 수행__ 해보고자 합니다.

## Genetic Algorithm
유전 알고리즘은 변수 선택 기법 중 가장 우수한 방법입니다. 이전까지의 변수 선택 기법들은 탐색 소요 시간을 줄여 효율적인 방법론을 제안하였으나, 탐색 범위가 적어 Global Optimum을 찾을 확률이 적은 한계를 가지고 있었습니다. 그러나 __자연계의 진화 체계를 모방한 메타 휴리스틱 알고리즘__ 인 GA는 시행착오를 통해 최적의 해를 찾아나가는 방법론으로, 다윈의 자연 선택설에 기반하여 초기에 다양한 유전자를 가지고 있던 종이 생존에 유리한 유전자를 택하면서 현재 상태가 되었다는 이론을 따라 해를 최적화 해나갑니다.
> **Heuristic 휴리스틱**   
> 참고로 휴리스틱이란 불충분한 시간이나 정보로 인하여 합리적인 판단을 할 수 없거나, 체계적이면서 합리적인 판단이 굳이 필요하지 않은 상황에서 사람들이 빠르게 사용할 수 있게 보다 용이하게 구성된 간편추론 방법론을 의미합니다. **메타 휴리스틱(Meta Heuristic)** 은 휴리스틱 방법론 중 풀이 과정 등이 구조적으로 잘 정의되어 있어 대부분의 문제에 어려움 없이 적용할 수 있는 휴리스틱을 의미합니다.

### 기본 개념
유전 알고리즘은 기본적으로 여러 개의 해로 구성된 잠재 해 집단을 만들고 적합도(fitness)를 평가한 뒤, 좋은 해를 선별해서 새로운 해 집단(후기 세대)을 만드는 메타 휴리스틱 알고리즘입니다. 작동 과정을 개략적으로 설명하면 아래와 같습니다.   
아래와 같이 함수 f(x)가 있고, x에 따른 f(x)의 최소값을 찾는 문제가 있다고 가정해봅시다. 

### Requirements
- Python >= 3.6
- numpy >= 1.18
- pandas >= 1.0.1
- rich >= 12.6.0

## References
- R. Tolosana, J.C. Ruiz-Garcia, R. Vera-Rodriguez, J. Herreros-Rodriguez, S. Romero-Tapiador, A. Morales and J. Fierrez, "Child-Computer Interaction: Recent Works, New Dataset, and Age Detection", IEEE Transactions on Emerging Topics in Computing, doi: 10.1109/TETC.2022.3150836, 2022.

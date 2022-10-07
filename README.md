# Business Analytics Topic1: Dimensionality Reduction (차원축소)
이미지, 텍스트, 센서 등 다양한 도메인의 데이터들은 변수의 수가 매우 많은 고차원 데이터(High Dimensional Data)의 특징을 가지고 있습니다. 그러나 많은 기계학습 알고리즘은 실제 데이터 차원을 모두 사용하지 않고, 정보를 축약하여 내재된 차원(Intrinsic/Embedded Dimension)으로 축약하는 경우가 많습니다. 이는 차원의 저주(curse of Dimensionality) 를 해결하기 위함인데, 사용하는 변수 수를 줄이면 잡음(noise)이 포함될 확률도 감소시킴과 동시에 예측 모델의 성능을 높이고, 예측 모델의 학습과 인식 속도를 빠르게 할 수 있으며 예측 모델에 필요한 학습 집합의 크기를 크게 할 수 있기 때문입니다. 따라서 분석 과정에서 성능을 저하시키지 않는 최소한의 변수 집합을 판별하여 주요 정보만을 보존하는 것입니다.
<br/>
본 튜토리얼에서는 차원 축소 방식 중 __비교사적 차원 축소(Unsupervised Dimensionality Reduction)__ 에 대한 4가지 방법론을 비교하여 설명하고자 합니다. 비교사적 차원 축소는 축소된 차원의 적합성을 검증하는 데 있어 예측 모델을 적용하지 않는다는 특징을 가집니다. 대표적 방법론으로는 50년 전 발표된 임베딩 방법론인 Multi-Dimensional Scaling(1964)과 가장 가까운 이웃(nearest neighbors) 정보를 이용하여 고차원 공간의 구조를 보존하는 저차원 공간을 학습하는 Locally Linear Embedding(2000), ISOMAP(2000), 그리고 안정적인 성능을 자랑하고 임베딩 결과 시각화에 가장 자주 사용되는 t-SNE(Stochastic Neighbor Embedding)가 있습니다. 사실 LLE 나 ISOMAP 은 deep learning 모델들의 임베딩 공간을 시각화 하기보다는 swissroll data 와 같이 manifold 가 존재하는 데이터 공간을 시각화 하는 데 적합합니다. 본 튜토리얼에서는 MDS, LLE, ISOMAP, t-SNE 임베딩 알고리즘들이 각자 차원을 축소하려했던 정보가 무엇인지 비교해 보려 합니다.
<br/>
고차원 공간의 벡터를 저차원으로 압축하여 시각화 할 때 중요한 정보는 가까이 위치한 점들 간의 구조입니다. 우리는 공간을 이해할 때 전체를 한 번에 인식하지 않습니다. 지구의 지도를 볼 때 특정한 지역에 집중합니다. 예를 들어 동아시아에서의 한국의 위치를 살펴본다면 한국의 우측에 일본이, 좌측에 서해를 사이에 둔 중국, 그리고 러시아 남부 일부가 눈에 들어옵니다. 이 때 우리는 한국과 인접한 정보들에 집중하며, 500 km 정도 떨어진 이 지역들이 한국을 중심으로 오른쪽에 있는지 왼쪽에 있는지가 중요합니다. 이 때, 남아메리카의 브라질이 한국과 얼마나 떨어져 있는지 아르헨티나가 얼마나 떨어지 있는지는 중요하지 않습니다. 둘 모두 대략 남서쪽 방향의 어딘가에 있다는 정보면 충분합니다. 그리고 유럽권 나라들은 서쪽 혹은 약간의 서북쪽 방향 어딘가에 있다는 정보면 충분합니다. 멀리 떨어진 점들의 정보는 디테일하게 집중할 필요가 없습니다. 하지만 locality 는 제대로 보존되어야 합니다. 이후 살펴볼 네 종류의 임베딩 알고리즘인 MDS, LLE, ISOMAP, t-SNE 을 이 관점에서 살펴보면 왜 t-SNE 가 고차원 벡터의 시각화 측면에서 안정적인 성능을 보이는지 이해할 수 있습니다.

## Multi-Dimensional Scaling (MDS)
Multi-Dimensional Scaling (MDS) 는 1964 년에 제안된, 매우 오래된 임베딩 방법입니다. MDS 는 원 공간 x 에서 모든 점들 간에 정의된 거리 행렬 δ 가 주어졌을 때, 

# Quiz

## 1. O/X 문제, 설명 추가해주세요
(1)

Bagging은 데이터에 랜덤성을 주어 여러 모델을 학습시키고,
aggregation을 통해 분산을 줄이는 앙상블 방법이다.

(2)

Random Forest에서 feature randomness란
bootstrap으로 데이터를 다시 뽑는 것을 의미한다.

(3)

Decision Tree에서 불순도(impurity)가 높을수록
예측이 쉬운 상태를 의미한다.

(4)

GBM은 이전 모델의 예측 오차(residual)를 그대로 다음 모델이 학습한다.

(5)

CatBoost가 boosting 모델인 이유는
이전 모델의 gradient를 근사하는 트리를 순차적으로 더하기 때문이다.

## 2. 다음 중 **Gradient Boosting Machine(GBM)**의 특성으로 가장 정확한 설명은 무엇인가?

(1)

GBM은 여러 약한 모델을 병렬로 학습시키고,
각 모델의 예측을 평균내어 분산을 줄이는 방법이다.

(2)

GBM은 이전 모델의 예측값을 고정한 채,
손실함수의 **gradient(의사 잔차)**를 잘 근사하는 모델을
순차적으로 추가하는 가산 모델이다.

(3)

GBM은 bootstrap과 feature randomness를 동시에 사용하여
트리 간 상관성을 줄이는 것을 핵심으로 한다.

(4)

GBM은 **각 트리**를 학습할 때
불순도(gini, entropy)를 직접 최소화하는 것을
주된 학습 목표로 한다.




## 정답
(1)

O (맞다)

Bagging은 데이터에 랜덤성을 주어 여러 모델을 학습시키고,
aggregation을 통해 분산을 줄이는 앙상블 방법이다.

(2)

X (틀림)

Random Forest에서 feature randomness란
bootstrap으로 데이터를 다시 뽑는 것을 의미한다(bagging 의 정의).

* 올바른 정답 : class를 랜덤샘플 하여 tree diversity를 확보

(3)

X (틀리다)

Decision Tree에서 불순도(impurity)가 높을수록
예측이 쉬운 상태를 의미한다.

* 올바른 정답 : 불순도가 높을수록 예측이 어려운 상태를 의미한다.

(4)

X (부분적으로만 맞아서 틀림)

GBM은 이전 모델의 gradient를 그대로 다음 모델이 학습한다.
* 부분 정답: residual 아니고 gradient (회귀에서는 같으나 분류에선 다름)

(5)

O (맞다)

CatBoost가 boosting 모델인 이유는
이전 모델의 gradient를 근사하는 트리를 순차적으로 더하기 때문이다.


---
2. 정답: ②

* GBM은 이전 모델의 예측값을 고정한 채,
손실함수의 gradient(의사 잔차)를 잘 근사하는 모델을
순차적으로 추가하는 가산 모델이다.

1. gbm 병렬학습 시키지 않음 (순차적) 이후는 맞는 설명
2. 맞음
3. bootstrap + feature randomness 는 random forest 의 특성
    * 틀린 방향을 따라서 모델을 더해서 보정시키는게 GBM 목적
4. 트리를 학습할때 불순도 최소화 는 decision tree의 특징이고, GBM 계열은 트리를 사용한 앙상블 모델인데, loss함수를 최소화하는게 목적 (gradient의 개념)
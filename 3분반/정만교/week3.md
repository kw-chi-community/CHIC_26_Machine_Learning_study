1. github 소개 영상
2. 케글데이터 사용금지 : 프로젝트
3. 현존 ai가 할수없는거 : 이상민 교수님 랩사이트 ref 해서, textmining 주제
`https://sites.google.com/view/aiaas`

정의, 장/단점
# week3 : 선형회귀와 분류



## Random Error : 입실론
train/val 을 어떻게 나누던 e 의 분포는 다음과 같다
* E(e) = 0
* Var(e) = σ²
* e ~ N(0, σ²)

### 모델의 기대값

기대값 
$$
\begin{aligned}
E(Y_i) &= E(\beta_0 + \beta_1 X_i + \epsilon_i) \\
&= \beta_0 + \beta_1 X_i + E(\epsilon_i) \\
&= \beta_0 + \beta_1 X_i
\end{aligned}
$$

### 모델의 분산
분산
$$
\begin{aligned}
Var(Y_i) &= Var(\beta_0 + \beta_1 X_i + \epsilon_i)
\end{aligned}
$$

여기서 중요한 전제:
* $X_i$는 고정값(fixed)으로 취급
* $\beta_0, \beta_1$는 상수

그래서:
$$
Var(\beta_0 + \beta_1 X_i) = 0
$$

결과적으로:
$$
Var(Y_i) = Var(\epsilon_i) = \sigma^2
$$

### **의미**
* 모델의 불확실성은 전부 오차항에서 온다










## keyword
- Loss Function
- LR(linear regression), Logit(logistic regression), SVM(support vector machine)
- Ridge, Lasso, Elastic Net 
- Bias-Variance trade-off

## Loss function
* 로스 함수
> 문제를 정의하고 ML / DL 모델을 통해 최적화 하는 것

* MSE : mean squared error (평균 제곱 오차)  
$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

* `실제값-예측값`의 제곱의 평균 
 
>$SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$  
>$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
* SSE 를 평균 낸 값
* ML/DL 에서는 SSE 보단 MSE를 많이 사용
    * 데이터 크기에 독립
    * gradient 관점: batch 크기가 작아서 학습 안정성이 높다

## linear regression
* 선형회귀
* data 들을 가장 잘 설명하는 함수를 찾는 것
> univariate LR  
> Multivariate LR  
![linear regression](res/1.png)  

* underfitting  
* overfitting

## logistic regression
* 로지스틱 회귀 : 선형결합을 확률로 변환해서 이진분류를 수행하는 선형 분류 모델
* classification 모델 (0 or 1)
* 핵심 함수: 시그모이드
![](../res/sigmoid.png)

| 항목   | 선형회귀 | 로지스틱 회귀       |
| ---- | ---- | ------------- |
| 문제   | 회귀   | 분류            |
| 입력 | 선형결합 / vector | 선형결합 / vector |
| 함수   | 항등식   | 시그모이드         |
| 출력   | 실수   | 확률 [0,1]         |
| 해석   | 단일 값 예측 | 확률에 따른 분류     |

> 강아지 고양이 분류 예시

![](res/sigmoid2.png)

1. 이미지 input
2. SVD 통해서 선형 차원 축소 &rarr; (wᵀx)
3. 각 feature(귀 모양, 코...)에 주어진 가중치의 선형결합을 합산 &rarr; (wᵀx + b)
4. 시그모이드 함수로 확률에 따른 분류


---

## SVM : Support Vector Machine
![](res/svm.png)

* 클래스 간 경계를 나누는 모델
    * 클래스는 feature 공유 집단
> 두 클래스 사이의 ‘여유 공간(마진, margin)’을 최대화하는 선  

1. 두 클래스 사이에 놓일 수 있는 **모든 가능한 선(초평면)** 을 고려
2. 각 선에 대해
3. 가장 가까운 데이터 점까지의 거리(마진)를 계산
4. 그 최소 거리(마진)가 최대가 되는 선을 선택

5. 이때 가장 가까운 점들이 바로 `support vectors`

![](res/svm2.png)

Loss f : hinge function  
(의미: support vectors 와 선의 거리를 최대화)  
* argmin 일텐데 의미적인 부분에서 설명한 것



---

## Regularization(규제)
### Ridge, Lasso, Elastic Net
> Loss Function에 “벌점(penalty)”을 추가하는 것
![](res/regularization.png)

### Lasso (L1 규제)
* cost f 에 가중치의 절대값의 합을 추가
* wj : 모델 파라미터(계수)
* 𝜆 : 규제 강도 (클수록 규제 강함)
>L1 규제는 “계수를 조금이라도 쓰면 비용이 든다”는 규칙을 만들고
그 비용을 감당할 만큼 도움이 안 되는 변수는
차라리 0으로 만드는 게 최적이 되게 설계되어 있다.
* model 을 학습시키면 cost 함수를 줄이는 방향으로 학습되기에 필요없는 변수는 사용하지 않는 현상 일어남(최적화) 

### Ridge (L2 규제)
* cost f 에 가중치의 제곱의 합을 추가
* wj : 모델 파라미터(계수)
* 𝜆 : 규제 강도 (클수록 규제 강함)
> “모든 가중치를 조금씩만 써라”  
큰 가중치 → 큰 벌점  
작은 가중치 → 거의 벌점 없음
#### w = 0 을 만들지 않음
* cost 를 미분해서 해를 구할때 w=0 이면 
![](res/l2.png) 이므로 w=0 이라면 의미 없는 식이 됨
* "각 변수를 조금씩만 사용해라" 라는 결론이 나게 됨

### Elastic Net
* Ridge와 Lasso의 조합
#### L1(Lasso)의 문제
* 변수 선택은 잘함
* 상관된 변수들 중 하나만 남기고 나머지는 버림
* 어떤 변수가 남을지 불안정
#### L2(Ridge)의 문제
* 안정적
* 하지만 변수를 안 버림
* 해석력 떨어짐
#### Elastic Net 필요성
![](res/elastic.png)
> 변수는 가능하면 적게 써라 (L1)
> 극단적으로 한 개에 몰지 말고
> 여러 개에 나눠 써라 (L2) 라는 의미

| 구분        | **L1 규제 (Lasso)** | **L2 규제 (Ridge)** | **Elastic Net**          |              |              |     |                         |
| --------- | ----------------- | ----------------- | ------------------------ | ------------ | ------------ | --- | ----------------------- |
| 기본 아이디어   | 불필요한 변수 제거        | 가중치 크기 억제         | 선택 + 안정성                 |              |              |     |                         |
| 계수 0 가능성  | 높음 (정확히 0)    | **거의 없음**             | 있음                       |              |              |     |                         |
| 변수 선택     | **가능**            | 불가능               | 가능                       |              |              |     |                         |
| 가중치 분배    | 한두 개에 집중          | 여러 변수에 분산         | 균형                       |              |              |     |                         |
| 상관된 변수 처리 | 하나만 남김 (불안정)      | 함께 유지             | **함께 유지 (group effect)** |
| 모델 안정성    | 낮음                | **높음**            | 중간~높음                    |
| 계산 안정성    | 보통                | **매우 좋음**         | 좋음                       |
| 주 사용 목적   | feature selection | overfitting 방지    | 고차원 + 상관 변수              |



## bias variance trade-off
$$ MSE = \underbrace{\text{Bias}^2}_{\text{단순함의 대가}} + \underbrace{\text{Variance}}_{\text{민감함의 대가}} + \underbrace{\text{Noise}}_{\text{줄일 수 없음}} $$


* 쉽게 말하자면 사실 `good fit / robust`한 model 을 만드는 것
* 좋은 model 이란 `mse`가 낮다 라고 말할 수 있음

#### 정의
- **Bias**: 모델의 *평균 예측값*과 *실제 함수* 사이의 차이  
  $$
  \text{Bias} = \mathbb{E}[\hat y] - y
  $$

- **Variance**: 학습 데이터가 달라질 때 *예측값*이 얼마나 변하는지  
  $$
  \text{Variance} = \mathbb{E}\big[(\hat y - \mathbb{E}[\hat y])^2\big]
  $$

#### Underfitting vs Overfitting
>underfitting : bias 높음, variance 낮음  
* bias :모델이 단순해서 오차가 큼
* variance : 제대로 예측하지 못해서 예측값의 고만고만함
>overfitting : bias 낮음, variance 높음  
* bias : 학습 데이터에 너무 맞춰져서 오차가 작음
* variance : 데이터 값이 조금만 달라져도 예측이 크게 달라짐

> bias 가 높다 &rarr; 모델이 멍청하다(underfitting)  
> variance 가 높다 &rarr; 모델이 너무 예민하다(overfitting)

#### 일반화 성능 모델
`Robust` 하다, `Good fit` 하다 라고 표현
* 모델 학습한 데이터와 실제 데이터는 차이가 있음
* 현실에서 얻는 데이터는 noise가 많이 포함되어 있음
* 데이터가 조금만 달라져도 예측이 크게 달라지지 않는 모델

* underfitting(bias) 과 overfitting(variance) 의 중간 지점을 찾아서 일반화가 잘되는 모델을 만드는 것이 중요하다 


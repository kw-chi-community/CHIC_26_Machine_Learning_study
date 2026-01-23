# Week 3 Quiz

### 1. 다음 중 **error($\epsilon$)와 residual($r$)**에 대한 설명으로 가장 적절한 것은?

① Error는 train/test 데이터에서 계산되는 오차이며, residual은 실전 데이터에서만 계산된다.

② Residual은 진짜 함수 $f(x)$를 기준으로 계산되며, error는 추정된 모델 $\hat{f}(x)$를 기준으로 계산된다.

③ **Error는 데이터 생성 과정에서 정의되는 확률변수로 관측할 수 없고, residual은 추정된 모델을 기준으로 계산되는 관측 가능한 값이다.**

④ Error는 residual이 허용 범위를 벗어났을 때 발생하며, 허용 범위 안에서는 residual만 존재한다.

⑤ Error와 residual은 동일한 개념이며, 표현만 다를 뿐이다.

<br>

### 2. Bias–Variance–$R^2$에 대한 설명으로 옳지 않은 것은?

① Underfitting 모델은 bias가 높고 variance가 낮으며, train과 validation의 $R^2$가 모두 낮다.

② Overfitting 모델은 train $R^2$는 높지만 validation $R^2$가 크게 감소하는 특징을 가진다.

③ Variance는 학습 데이터가 달라질 때 예측값 $\hat{y}$가 얼마나 변하는지를 의미한다.

④ **$R^2$ 값이 높을수록 항상 일반화 성능이 좋은 모델이라고 할 수 있다.**

⑤ Adjusted $R^2$는 파라미터 수 증가에 따른 $R^2$의 과대평가를 보정하기 위해 사용된다.

---

## 정답 및 해설

**1. 정답: ③**

> **해설**
> * **Error($\epsilon$)**: 데이터 생성 과정의 확률변수 ($y = f(x) + \epsilon$). $f(x)$를 알 수 없으므로 분리하거나 관측할 수 없음.
> * **Residual($r$)**: 잔차, $r = y - \hat{f}(x)$로 항상 계산 가능.
> * 파란 허용 범위 안/밖은 residual의 크기 차이이지 error 여부가 아님.

<br>

**2. 정답: ④**

> **해설**
> * $R^2$는 설명력 지표이지 일반화 성능 지표가 아닙니다.
> * Overfitting 모델은 train $R^2$가 매우 높아도 test 데이터에 대한 성능은 나쁠 수 있습니다.
> * 따라서 train–validation 간 $R^2$ 차이(Gap)가 핵심 판단 기준이 됩니다.
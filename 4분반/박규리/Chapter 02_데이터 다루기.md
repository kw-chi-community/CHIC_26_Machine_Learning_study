# 지도 학습
- 알고리즘이 정답을 맞히는 것을 학습
- 훈련하기 위한 훈련 데이터가 필요

# 비지도 학습
- 타깃 없이 입력 데이터만 사용
- 무언가를 맞힐 수는 없지만, 데이터를 잘 파악하거나 변형하는데 도움을 줌


# 훈련 세트와 테스트 세트
- 정확한 평가 -> 테스트 세트와 훈련 세트 따로 준비
## 훈련세트
모델을 훈련할 때 사용하는 데이터
## 테스트 세트
전체 데이터에서 20% ~ 30%만 사용하는 경우가 많음

# numpy
- 파이썬의 대표적인 배열 라이브러리
- `array()` 함수 활용하여 리스트를 배열로 변환
- 인덱스, 슬라이싱 기법으로 섞음

# 머신러닝 적용
[코랩 링크](https://colab.research.google.com/drive/1NmfyRcj7LbsJBN8xMbdVhPBdeRgje0o-#scrollTo=LQEfg_XBHbS9)

---
# 결측치 
: 수집된 데이터 셋 중 관측되지 않은 특정 확률변수의 값

## 종류
- MCAR : 결측치가 완전히 random으로 발생
- MAR : 결측치가 관측된 특정 변수로 추정 가능
- MNAR : 결측치가 random으로 발생 x, 관측값, 결측값 모두에게 영향을 받음

## 결측치 대체 방법
- 단순 & 평균 & 단순 확률 & 다중 대치법

# 데이터 분포 변형 방법론

## 표준화 
: 데이터가 평균으로부터 얼마나 떨어져서 분포하는지 표현하는 변환(표준정규분포)
- 선형회귀
- 로지스틱회귀
- 선형판별분석

## 정규화 
: 상대적 크기에 대한 영향을 줄이기 위한 변환
- MinMax 스케일링
- Robust 스케일링
- [0,1] 스케일링

# 범주형 변수
종류 - 명목형, 순서형

## 인코딩 방법
- One Hot Encoding
    - Cardinality(범주의 개수)가 작을 때 사용
    - 항상 직교하여 거리 정보가 의미 x
    - 희소행렬 문제가 발생
- Hashing Encoding
    - Hashing trick을 활용하여 인코딩 하는 방식
    - One-hot encoding 대비 더 적은 dummy variable 생성 가능
---
# 데이터 전처리
## 넘파이로 데이터 준비하기
```
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
```
```
import numpy as np
#column_stack() 함수 => 전달받은 리스트 일렬로 세우고 차례대로 나란히 연결
np.column_stack(([1,2,3], [4,5,6]))

#위와 같은 column_stack함수 이용하여 fish_length와 fish_weight 맵핑
fish_data = np.column_stack((fish_length, fish_weight))

#target맵핑도 [0]과 [1]여러번 곱해서 만들지 말고 np.ones()랑 np.zeros 이용해서 만듦
#데이터가 큰 경우 이렇게 해야 함
fish_target = np.concatenate((np.ones(35),np.zeros(14)))
print(fish_target)
```
## 사이킷런으로 훈련 세트와 테스트 세트 나누기
```
from sklearn.model_selection import train_test_split

#train)test_split()에는 자체적으로 랜덤시드 지정할 수 있는 random_state변수 있음
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state = 42)
#이렇게 하면 2개의 입력데이터와 2개의 타깃 데이터 => 총 4개의 배열 반환

#2차원 배열인 훈련 데이터
print(train_input.shape, test_input.shape)

#1차원 배열인 타겟 데이터
print(train_target.shape, test_target.shape)

#도미와 빙어가 잘 섞였는지 테스트 데이터 출력
print(test_target)

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify = fish_target,random_state = 42)

print(test_target)
```
## 수상한 도미 한 마리
```
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
#결과 1로 모델 성능 높음

#새로운 데이터에 정확한 결과X
print(kn.predict([[25,150]]))

import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker = '^') #marker로 매개변수 모양 지정 => 25,150은 세모로 표시
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#이웃값 반환 코드, kneighbors쓰면 n_neighbors의 기본값 5라서 매개변수의 5개 이웃 반환
distances, indexes = kn.kneighbors([[25,150]])

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker = '^') 
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker = 'D') #D는 마름모
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#타켓데이터로 이웃 생선의 4개가 빙어임을 확인
print(train_target[indexes])

#indexes가 아닌 distances배열 출력
print(distances)
```
## 기준을 맞춰라
```
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker = '^') 
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker = 'D')
plt.xlim((0,1000)) #x축 원래 0부터 40이었는데 범위 늘려서 x축의 차이가 이웃점 선택에 너무 큰 영향을 끼치지 않도록 설정
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
#즉, 두 특성의 스케일이 달라서 이를 맞춰줌

#표준점수(=z점수) : 평균에서 표준편차의 몇 배만큼 떨어져 있는지 나타냄 => 평균 빼고 표준편차를 나누면 됨 둘 다 넘파이에서 제공함
mean = np.mean(train_input, axis = 0) #평균
std = np.std(train_input, axis = 0) #표준 편차
print(mean, std)

#모든 trian_input의 행에서 mean에 있는 두 평균값 빼주고 std에 있는 두 표준편차 모든 행에 적용함 => 브로드캐스팅
train_scaled = (train_input - mean) / std
```
## 전처리 데이터로 모델 훈련하기
```
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker = '^') 
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker = 'D')
plt.xlim((0,1000)) #x축 원래 0부터 40이었는데 범위 늘려서 x축의 차이가 이웃점 선택에 너무 큰 영향을 끼치지 않도록 설정
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
#즉, 두 특성의 스케일이 달라서 이를 맞춰줌

#표준점수(=z점수) : 평균에서 표준편차의 몇 배만큼 떨어져 있는지 나타냄 => 평균 빼고 표준편차를 나누면 됨 둘 다 넘파이에서 제공함
mean = np.mean(train_input, axis = 0) #평균
std = np.std(train_input, axis = 0) #표준 편차
print(mean, std)

#모든 trian_input의 행에서 mean에 있는 두 평균값 빼주고 std에 있는 두 표준편차 모든 행에 적용함 => 브로드캐스팅
train_scaled = (train_input - mean) / std

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(25, 150, marker = '^') 
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
#샘플도 표준화 해줘야함을 보여줌

new = ([25,150] - mean) / std #샘플도 표준화
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker = '^') 
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

test_scaled = (test_input - mean) / std

kn.score(test_scaled, test_target)

print(kn.predict([new]))

distances, indexes = kn.kneighbors([new])

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker = '^') 
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker = 'D') #D는 마름모
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

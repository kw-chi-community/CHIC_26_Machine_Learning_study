# Week 6 비선형 모델

## Keyword
```
activation function, sigmoid, ReLU
ShallowNN, DeepNN, AE
RNN, LSTM, Seq2Seq
```

## RNN : Recurrent Neural Network
* Recurrent : 반복적인 , 되풀이되는 , 재발
* Neural Network : 신경망
> 순서가 있는 데이터를 처리하는 신경망

순서가 있는 데이터 &rarr; 이전 상태(시간)가 다음 상태에 영향을 준다

```
끝말잇기 : 이전 단어(state)가 다음 단어(state)에 영향을 준다
```
* 조금 다른점 : 끝말잇기는 이전 단어에만 영향을 받지만 RNN은 과거에 있었던 모든 state(상태)를 input으로 받는다

![](./res/6/rnn.png)

### 상태 전이
![](./res/6/rnn2.png)

h(은닉층에서의 벡터) = x,y 의 state 들을 함수 f 에 넣은 값 

* RNN의 핵심 : 이전 state가 언제 다음 layer로 넘어가는가? &rarr; 초록 그림에서 input을 넣었을때 hidden layer 에서 다음 step의 hidden layer로 넘어감
#### 왜 output을 다음 input으로 전달하지 않는가?
* hidden layer로 전달하는게 벡터 이므로 더 많은 차원의 정보를 가지고 있음
* output은 softmax로 압축된 결과
* hidden layer &rarr; hidden layer가 온전한 정보 전달


### 종류
1. one to one
2. many to one 
    * input 이 여러개, 출력은 은닉층들을 거친 최종 결과 1개
3. one to many
    * 하나의 입력, 은닉층을 거칠때 마다 출력 (진행 상황을 보기 편함)
4. many to many
    * input을 매 스텝마다 받고, 그때마다 결과 출력함





## LSTM : Long Short Term Memory


## Seq2Seq : Sequence to Sequence
* 

## Transformer : Attention is all you need
* google 에서 발표한 `attention is all you need ` 논문

원리 : self-attention



## 문1 : O/X 적고 설명 쓰시오
1. RNN에서 hidden state는 이전 출력값을 그대로 전달한 것이다. ( )

2. RNN은 모든 과거 state를 직접 입력으로 받는다. ( )

3. LSTM의 cell state는 gradient vanishing 문제를 완화하기 위해 덧셈 구조를 사용한다. ( )

4. Seq2Seq는 입력 길이와 출력 길이가 항상 동일하다. ( )

## 문2 : 서술형
1. 문장 길이가 5인 경우와 50인 경우, RNN에서 학습 안정성 차이가 발생하는 이유를 설명하고 어떤 길이의 문장이 더 안정적으로 output을 가져오는지 말하시오. 


2. LSTM에서 cell state 𝐶(장기기억 항) 가 중요한 이유를 미분 관점에서 설명하시오.










## 답
문1:  
1. x : h 로 정보 압축하여 다음 hidden state 에 전달
2. x : 직접 입력 아니고 h로 받음
3. o : 덧셈 항으로 c를 따로 빼서 미분하여 chain rule 에 의한 gradient 소실을 완화
4. x : 번역처럼 m to n 이런 sequence를 처리하기 위해 만듦


문2:   
1. 5가 더 안정적. 안정성 차이가 발생하는 이유는 RNN은 gradient를 미분하여 구할때 가장 처음 들어왔던 state가 chain rule에 의해 0에 가까워져 기억의 소실이 일어남 (vanishing gradient)

2. c를 따로 빼서 추가하여 vanishing gradient를 막음
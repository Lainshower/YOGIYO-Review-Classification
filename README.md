# YOGIYO-Review-Classification
Term Project in Advanced Machine Learning

---
### Dataset
---
* 요기요에서 세종대학교 주변 음식점 69개에 대한 리뷰를 크롤링해서 데이터셋을 구축
* 정규표현식을 활용해서 특수기호 및 이모티콘을 제거했고, 네이버 맞춤법 검사기 바탕의 py-hanspell을 활용해 맞춤범 검사를 진행
* 자세한 데이터셋 구축 및 전처리는 [preprocessing/전처리과정.ipynb](https://github.com/Lainshower/YOGIYO-Review-Classification/blob/main/preprocessing/%EC%A0%84%EC%B2%98%EB%A6%AC%EA%B3%BC%EC%A0%95.ipynb)을 참고하시면 됩니다.

* 아래와같이 '맛', '배달', '양'과 관련된 형태소가 없는 경우, 텍스트 기반 별점추론이 매우 어려워 Rule-based 방법을 도입해 '맛', '배달', '양'과 관련된 형태소가 존재하는 리뷰들만 필터링해 모델링을 하기로 결정

* '맛', '배달', '양'과 관련된 형태소가 없는 리뷰의 예시

|맛|배달|양|리뷰|
|:---:|:---:|:---:|:---:|
|3|3|2|주꾸미가 몇 개 없는데 다 아기들이네요. ᅲᅲ|
|3|3|3|그냥 그래요....|
|5|5|5|앞으로 여기서 자주 시켜 먹을 듯 너무 좋았다 진짜로|
|3|2|4|정식이 두 개인데 국물은 하나만 와서 전화 여러 번 했는데 안 받으시네요 앞으로 꼼꼼히 좀 챙겨주세요.|
|1|5|1|전에도 그러시더니 또 비빔장에 고추장을 안 주셨어요 ᅲ.ᅲ 이제 여기서 안 시켜 먹을래요.|


---
### Model
---
#### Machine Learning
  - Embedding : Pretrained Fasttext Embedding from [Fasttext](https://fasttext.cc/docs/en/support.html)
  1. Random Forest
  2. SGD Classifier
  3. XGBoost
  
#### Pre-trained Model
  - BERT
  
---
### Results
---
  #### Quantitative Results
  * XGBoost
  <p align="center">
   <img src = "https://github.com/Lainshower/YOGIYO-Review-Classification/blob/main/results/xgboost-result.png?raw=true">
  </p>
  
  * BERT
  <p align="center">
   <img src = "https://github.com/Lainshower/YOGIYO-Review-Classification/blob/main/results/BERT-result.png?raw=true">
  </p>

  #### Qualitative Results
  * XGBoost

  |항목|실제|예측|리뷰|
  |:---:|:---:|:---:|:---:|
  |맛|3|1|햄버거가 .. 점점 맛이 없어져요.|
  |맛|4|5|처음 먹어봤는데 맛있어요!|
  |배달|3|1|배달시간 오래 걸리는 거 빼곤 조음|
  |배달|2|3|맛있게 잘 먹었습니다. 근데 좀 늦네용|
  |양|2|3|맛은 그럭저럭. 양은 흐 으음...... 배달은 빠름.|
  |양|4|3|프라이드는 역시 이집인 듯 요ᄒ양이 살짝 적은 거 같지만 잘 먹겠습니다|

  * BERT

  |항목|실제|예측|리뷰|
  |:---:|:---:|:---:|:---:|
  |맛|5|3|이게 무슨 피자죠. 맛이나 양은 그렇다 치고 배달이 문젠지 잘못 만든 건지...|
  |맛|3|4|맛있어요. 배달도 빠르고 좋아요. 근데 짜장면 면은 별로입니다. 땡글땡글한 맛은 없어요.|
  |배달|2|3|맛있게 잘 먹었습니다. 배달이 잘못 오고 또 늦게 와서.... 좀 아쉬웠지만 치킨은 맛있네요 ᄒᄒ|
  |배달|5|3|배달이 좀 걸리네요.. 잘 먹었습니다|
  |양|2|3|대짜리 시켰는데.. 양도 너무 적고 젓가락도 2개밖이 안 주시고....|
  |양|3|4|오또케 너무 마싯자낭 그래서 양이 부족해썽ᅲᅲ|

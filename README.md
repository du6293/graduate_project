# graduate_project
# 작품명 : 딥러닝을 사용한 한우 중량 측정

# 작품 설명 영상 : https://youtu.be/6KjvvEqXCNw]

1.데이터 배분 및 라벨링작업

실험에 이용된 한우는 모두 강원도 평창에 있는 농촌진흥청 국립 축산과학원 한우연구소를 통해서 Top View이미지로전달 받았고, 총 개체 수는 46마리의 영상을 받아 영상의 프레임별로 100장씩 나누어 소의 이미지를 얻어냈다.
Instance Segmentataion 방법으로 한우의 영역 검출을 하기 위해, 소의 TOP VIEW를 VIA(vgg image annotator)를 사용하여 학습 데이터셋에 Annotation 작업을 하였다. 

![1](https://user-images.githubusercontent.com/76850241/182006994-52846ac2-5bbc-4d40-9679-8209bc9502ed.PNG)

![1](https://user-images.githubusercontent.com/76850241/182006960-071fdb49-bf84-4cf1-b9a3-0bdd5bceef87.PNG)

![KakaoTalk_20220730_223317077](https://user-images.githubusercontent.com/76850241/181916846-9028fb9d-cd6d-4341-87d4-c9d88df34ab4.jpg)


2.Mask R-CNN을 통한 객체 검출
Annotation 작업이 끝난 데이터셋들을 분류하기 위해서 Mask R-cnn의 기능 중 instance segmentation task중에서 instance segmentation을 사용하여 객체단위로 분류를 진행해 주었고, instance를 픽셀 단위로 분류하는 task를 해주었다.

![q](https://user-images.githubusercontent.com/76850241/182008917-abcf11e2-2442-490c-920e-d677fb708310.PNG)

 Mask 영상 획득
from google.colab import drive
drive.mount('/gdrive')

!git clone https://github.com/matterport/Mask_RCNN '/gdrive/My Drive/Colab Notebooks/Mask_RCNN'
!pip install tensorboard==1.15.0 tensorflow==1.15.0 tensorflow-estimator==1.15.1 tensorflow-gpu==1.15.2 tensorflow-gpu-estimator==2.1.0 Keras==2.2.5 
!python3 '/gdrive/My Drive/Colab Notebooks/Mask_RCNN/samples/balloon/balloon.py' train --dataset='/gdrive/My Drive/Colab Notebooks/Mask_RCNN/samples/balloon/cow_


3. opencv로 한우의 top view를 타원형으로 검출
![KakaoTalk_20220727_212523070_05](https://user-images.githubusercontent.com/76850241/181916423-fc227880-6a05-4c8c-882b-b46be81174e2.jpg)
![KakaoTalk_20220727_212523070_08](https://user-images.githubusercontent.com/76850241/181916428-7b0cbc97-3c40-49aa-83bd-a385e5198dbe.jpg)

4. 입력값을 다르게하여 성능 개선

![qq](https://user-images.githubusercontent.com/76850241/182008901-9c5145a5-eb36-4a31-8ccb-1dcd067bdefc.PNG)


Mask R-CNN을 이용하여 Mask를 얻은 뒤, OpenCV를 활용하여 입력값으로 사용할 정보들을 추출해내었다. 입력값으로는 
1. ‘Mask의 pixel 개수’ 
2. ‘Mask의 pixel 개수 & 타원 넓이’  
3. ‘Mask의 pixel 개수 & 장축 및 단축’을 사용하였다. 


예측에 사용한 네트워크 구성에 대해서 살펴보자면, 평가 지표는 MAE(Mean Absolute Error)를 사용하였다. Learning Rate는 1e-6, Epochs는 EarlyStop을 사용하여 MAE를 monitor하여 30Epoch동안 개선이 없으면 학습을 중단시켰다. Optimizer는 RMSprop를 사용하였으며, Input layer와 두 개의 Dense layer, Output layer로 구성되어 있다. 

Mask의 pixel개수만 사용한 경우, MAE값이 27로 pixel개수만으로는 유의미한 예측 결과를 얻기 힘들었다. 따라서 Mask타원의 넓이를 입력 값에 추가해 본 결과 MAE가 20이 나왔다. 이 또한 예측 모델로 사용하기에는 어려운 결과 값이었기 때문에 Mask의 장축과 단축을 입력 값에 추가해주었다. 총 4개의 변수를 Input layer에 넣어 MAE값 16을 얻을 수 있었다. fig. 3는 Loss Function MSE와 평가 지표 MAE가 Epoch이 진행됨에 따라 값이 변하는 모습을 보여준다. 

5.train data와 test data비교

![2](https://user-images.githubusercontent.com/76850241/182006969-dc795f85-d9d8-4583-b551-5db356ddb90b.PNG)
![KakaoTalk_20220727_212523070_02](https://user-images.githubusercontent.com/76850241/181916432-ec6c304d-6d13-481f-9bf2-b65d04c60d7b.jpg)
![KakaoTalk_20220727_212523070_04](https://user-images.githubusercontent.com/76850241/181916434-9e4bd606-573d-4355-9316-1d9ced9442d8.jpg)


# 결론

영상처리 기술을 축산업에 접목시켜 가축의 중량을 추정하는 방법이 연구되고 있다.
본 연구는 Mask R-CNN영상을 바탕으로 OpenCV기법을 접목시켜 한우의 무게를 추정하기 위해 수행되었다. 강원도 평창 농촌진흥청 국립 축산과학원 한우 연구소에서 46마리 소의 Top View 영상을 받아 프레임별로 100장씩 나누어 이미지를 얻어내었다.
소의 Top View를 VIA(vgg image annotator)를 사용해 Annotation 작업을 한 후, instance segmentation을 사용하여 객체단위로 분류작업을 진행하고 픽셀단위로 분류하는 task를 하였다. 이미지 분할 알고리즘인 Mask R-CNN을 이용해 Mask영상을 얻고, OpenCV기법을 활용하였다.  Mask 영상을 얻는 과정에서 한우의 꼬리부분은 정밀하게 분류가 되지 않아 실제 소의 Top View면적보다 면적값이 적게 나왔다. 소의 중량을 예측하기 위해 사용한 parameter은 다음과 같다. 예측에 사용된 네트워크 평가지표는 MAE(Mean Absolute Error)를 사용하였고, Learning Rate는 1e-6, Epochs는 Early Stop, Epoch은 30을 주어 학습을 진행했다. Optimizer로는 RMSprop을 사용하였고, Input layer와 두 개의 Dense layer, Output layer로 구성하였으며, 손실함수로는 MSE(Mean Squared Error)를 사용했다.
 3가지의 중량추정 방법에서 Mask pixel 개수와 장축과 단축의 길이를 사용하였을 때 가장 정확한 추정이 가능하였다. Mask pixel 개수만 사용하였을 때 MAE값이 27로 나와 유의미한 예측값을 얻기 어려워, Mask의 pixel개수와 타원의 넓이를 함께 사용한 경우 MAE값이 20이 나왔다. 보다 정확한 무게 추정을 위해 장축과 단축의 길이를 함께 사용한 경우 MAE값이 16까지 떨어진 모습을 확인할 수 있었다. 사용된 46마리의 소의 평균 중량은 340.0435kg, 분산은 22448.61로, 표준편차는 149.8286으로, 평균 중량보다 무게가 덜 나가는 개체들이 상당수 있었다. 차후에 더 많은 개체수의 한우중량 데이터를 확보한다면 더 적은 MAE값을이용해 정확하게 중량추정을 할 수 있을 것이다.
결과적으로mask R-CNN 영상과 OpenCV기법을 이용해 무게를 추정하는 방법은 기존의 우형기를 이용한 방식보다 더 효율적이라고 볼 수 있다.

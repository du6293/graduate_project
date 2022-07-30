# graduate_project
# 작품명 : 딥러닝을 사용한 한우 중량 측정

# 작품 설명 영상
https://youtu.be/6KjvvEqXCNw]

1.데이터 배분 및 라벨링작업

![KakaoTalk_20220727_212523070_03](https://user-images.githubusercontent.com/76850241/181916385-f8fc2e54-75d0-4498-abe0-e7311852d83c.jpg)
![KakaoTalk_20220730_223317077](https://user-images.githubusercontent.com/76850241/181916846-9028fb9d-cd6d-4341-87d4-c9d88df34ab4.jpg)

2.Mask영상획득

## Mask 영상 획득
from google.colab import drive
drive.mount('/gdrive')

!git clone https://github.com/matterport/Mask_RCNN '/gdrive/My Drive/Colab Notebooks/Mask_RCNN'
!pip install tensorboard==1.15.0 tensorflow==1.15.0 tensorflow-estimator==1.15.1 tensorflow-gpu==1.15.2 tensorflow-gpu-estimator==2.1.0 Keras==2.2.5 
!python3 '/gdrive/My Drive/Colab Notebooks/Mask_RCNN/samples/balloon/balloon.py' train --dataset='/gdrive/My Drive/Colab Notebooks/Mask_RCNN/samples/balloon/cow_

3.opencv로 한우의 top view를 타원형으로 검출
![KakaoTalk_20220727_212523070_05](https://user-images.githubusercontent.com/76850241/181916423-fc227880-6a05-4c8c-882b-b46be81174e2.jpg)
![KakaoTalk_20220727_212523070_08](https://user-images.githubusercontent.com/76850241/181916428-7b0cbc97-3c40-49aa-83bd-a385e5198dbe.jpg)


4.train data와 test data비교
![KakaoTalk_20220727_212523070_02](https://user-images.githubusercontent.com/76850241/181916432-ec6c304d-6d13-481f-9bf2-b65d04c60d7b.jpg)
![KakaoTalk_20220727_212523070_04](https://user-images.githubusercontent.com/76850241/181916434-9e4bd606-573d-4355-9316-1d9ced9442d8.jpg)

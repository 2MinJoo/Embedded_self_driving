# Embedded_self_driving

[![preview](https://img.youtube.com/vi/EZEgyinBsww/0.jpg)](https://youtu.be/EZEgyinBsww?t=0s)

## 라즈베리파이 + 파이카메라 + 파이썬 + openCV 실시간 영상처리

- 영상처리를 통한 자율주행 알고리즘 개발

- 2019.10 ~ 2020.08

- pi 카메라를 통해 실시간 임시 도로의 영상을 얻음
- 이미지의 RGB 데이터 값을 행렬 형태로 받아오게 되는데, OpenCV 필터링 작업과 임계치 작업을 통해 단순화된 흑백 이미지 형태로 변환
- 획득한 흑백 이미지 데이터값을 통해 기울기값, 현재 위치 값 등을 계산한 뒤 실시간으로 진행 방향을 계산
- 라즈베리파이 + python + OpenCV

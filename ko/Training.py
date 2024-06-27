import pandas as pd    # 데이터셋 처리
import joblib   # 훈련된 모델 로드 및 저장
from sklearn.datasets import fetch_openml   # 손글씨 숫자 이미지
from sklearn.model_selection import train_test_split    # 훈련 데이터와 테스트 데이터를 나누고 비율을 조정
from sklearn.neighbors import KNeighborsClassifier  # K-최근접 이웃 분류기, 즉 KNN 알고리즘, 가장 가까운 K개의 이웃을 기준으로 예측
from sklearn.metrics import accuracy_score  # 모델의 정확도 계산

mnist = fetch_openml('mnist_784', version=1)
# mnist_784라는 이름의 데이터셋을 가져오고 버전은 1입니다. 이 데이터셋에는 0~9까지의 손글씨 숫자 이미지 7만 개가 포함되어 있습니다.
# 흥미로운 배경으로, 이 데이터셋의 6만 개의 훈련 이미지는 미국 인구 조사국 직원들이 작성한 것이고, 1만 개의 테스트 이미지는 미국 고등학생들이 작성한 것입니다.

X = pd.DataFrame(mnist['data'])
# 이 데이터셋에서 특징을 추출합니다. 이 특징들은 이차원 벡터로 데이터셋에서 'data'로 인덱싱되어 있으므로 DataFrame으로 처리해야 합니다.
y = pd.Series(mnist.target).astype('int')
# 이 데이터셋의 레이블, 즉 28*28 이차원 벡터에 해당하는 목표 값은 일차원 벡터이므로 Series로 처리합니다.
# 목표 값이 문자열 유형으로 설계되어 있으므로 astype('int')를 사용하여 int 유형으로 변환해야 합니다.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# train_test_split 함수를 사용하여 특징과 레이블(X와 y)을 80%와 20% 비율로 나눕니다.
# 따라서 X_train과 y_train은 훈련에 사용되고, X_test와 y_test는 테스트에 사용됩니다.

estimator = KNeighborsClassifier(n_neighbors=3)
# K-최근접 이웃 분류기를 인스턴스화하고 n_neighbors를 3으로 설정합니다. 이는 미래의 데이터 테스트나 모델 사용 시
# 이 알고리즘이 특징 공간에서 가장 가까운 세 개의 이웃을 자동으로 검색하고(일반적으로 유클리드 거리 사용)
# 이 세 이웃의 카테고리를 기반으로 새로운 데이터 포인트의 카테고리를 결정한다는 것을 의미합니다.

estimator.fit(X_train, y_train)
# 이 분류기의 fit 메서드를 사용하여 모델 훈련을 시작하고 훈련 데이터와 해당 레이블을 전달합니다.
# 모델이 데이터 특징과 목표 값 간의 관계를 학습하고 기억하여 나중에 예측에 사용할 수 있게 합니다.

y_pred = estimator.predict(X_test)
# 모델 훈련이 완료된 후, 위에서 분할한 테스트 데이터를 사용하여 예측을 수행하고 모델의 정확도를 계산합니다.
# 분류기의 predict 메서드를 호출하고 테스트 데이터 X_test를 전달하면, 학습한 규칙에 따라 각 테스트 샘플에 대한 예측 결과를 반환합니다.

print(accuracy_score(y_test, y_pred))
# 테스트 결과를 비교하여 모델의 정확도를 계산합니다.

joblib.dump(estimator, '../mnist_784.pth')
# 마지막으로 joblib 라이브러리의 dump 메서드를 호출하여 훈련된 모델을 전달하고 파일 이름을 'mnist_784.pth'로 설정합니다.
# 이렇게 하면 모델이 디스크에 저장되고, 모델을 다시 훈련할 필요 없이 직접 로드하여 사용할 수 있습니다.

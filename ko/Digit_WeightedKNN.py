import warnings   # 경고 메시지 무시
import joblib   # 훈련된 모델 로드 및 저장
import numpy as np  # 배열 작업
from PIL import Image  # 이미지 파일 처리

class DigitRecognizer:
    # 손글씨 숫자 인식기 클래스를 생성

    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        # joblib 메서드를 사용하여 훈련된 모델을 로드
        self.X_model = self.model._fit_X
        self.y_model = self.model._y
        # 모델에서 훈련 데이터와 라벨을 추출

    def compute_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
        # 공식을 사용하여 두 점 사이의 유클리드 거리를 계산

    def compute_weight(self, distance):
        a, b = 1, 1
        # 가중치를 조정하기 위한 매개변수를 정의, a는 0으로 나누는 것을 피하기 위한 스무서로 이해할 수 있음
        return b / (distance + a)
        # 가중치 값을 계산하고 반환. 이 공식은 거리가 작을수록 가중치가 크고, 거리가 클수록 가중치가 작다는 것을 보여줌

    def predict_digit(self, filename):
        img = Image.open(filename).convert('L')
        img = img.resize((28, 28))
        img = np.array(img).reshape(1, -1)
        # 입력된 이미지를 처리. 이는 이전에 설명했으므로 반복하지 않겠음

        distances = []
        # 입력 이미지와 각 훈련 샘플의 거리와 라벨을 저장하기 위한 빈 리스트를 생성
        for i, X_train in enumerate(self.X_model):
            # 훈련 세트의 각 샘플을 반복 처리

            distance = self.compute_distance(img, X_train.reshape(1, -1))
            # compute_distance 함수를 사용하여 이미지와 훈련 샘플 사이의 거리를 계산. 훈련 샘플을 img의 형태에 맞게 리쉐이프하는 것을 잊지 말아야 함

            weight = self.compute_weight(distance)
            # 가중치 계산

            distances.append((weight, self.y_model[i]))
            # 가중치와 해당 라벨을 튜플로 리스트에 추가

        distances.sort(key=lambda x: x[0], reverse=True)
        # 람다 표현식을 사용하여 리스트를 가중치에 따라 내림차순으로 정렬

        k_neighbors = distances[:3]
        # 정렬 후 가중치가 가장 높은 세 이웃을 선택

        weighted_votes = {}
        # 각 라벨에 대한 가중치 투표 결과를 기록하기 위한 빈 사전 생성

        for weight, label in k_neighbors:
            # 세 이웃의 가중치와 라벨을 반복 처리

            if label in weighted_votes:
                weighted_votes[label] += weight
                # 라벨이 사전에 이미 존재하는 경우, 단순히 이 가중치를 누적하면 됨

            else:
                weighted_votes[label] = weight
                # 그렇지 않으면, 현재 가중치로 이 라벨에 대한 새로운 항목을 생성

        predictions = max(weighted_votes, key=weighted_votes.get)
        # 가중치 투표 결과에서 가장 높은 가중치를 가진 라벨을 최종 예측 결과로 선택

        return predictions
        # 이 결과를 반환

def digit_test():
    warnings.filterwarnings("ignore")
    # 훈련 시의 특징 이름과 현재의 특징 이름이 다르기 때문에 경고가 발생하지만, 실행 및 결과에는 영향을 미치지 않음
    # 따라서 이 경고를 무시하면 됨. 해결하려면 모델을 테스트하기 전에 다음 코드를 추가하면 됨
    # X.columns = [f'pixel{i}' for i in range(X.shape[1])]

    recognizer = DigitRecognizer('../mnist_784.pth')
    filename = '../handwritten_digits/digit_1.png'
    prediction = recognizer.predict_digit(filename)
    print(f'테스트 결과는: {prediction}')
    # 예측 이미지에 대해서는 이전 파일에서 설명했으므로 반복하지 않겠음

if __name__ == '__main__':  # 메인 함수
    digit_test()

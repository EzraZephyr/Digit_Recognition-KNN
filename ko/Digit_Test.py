import warnings   # 경고 메시지 무시
import joblib   # 훈련된 모델 로드 및 저장
import numpy as np  # 배열 작업
from PIL import Image  # 이미지 파일 처리

def digit_test():
    warnings.filterwarnings("ignore")
    # 훈련 시의 특징 이름과 현재 특징 이름이 달라 경고가 발생하지만 실행 및 결과에는 영향을 미치지 않습니다
    # 따라서 이 경고를 무시하십시오. 해결하려면 모델을 테스트하기 전에 다음 코드 줄을 추가하십시오
    # X.columns = [f'pixel{i}' for i in range(X.shape[1])]

    model = joblib.load('../mnist_784.pth')
    # joblib 메서드를 사용하여 훈련된 모델을 로드합니다

    filename = '../handwritten_digits/digit_1.png'
    # 이미지 파일의 경로를 filename에 저장하여 아래에서 직접 호출할 수 있도록 합니다

    img = Image.open(filename).convert('L')
    # Image.open 메서드를 사용하여 해당 경로의 이미지를 열고 convert 메서드를 사용하여 그레이스케일로 변환합니다

    img = img.resize((28, 28))
    # 이미지 크기를 28*28 형식으로 압축하여 모델의 입력 요구 사항을 충족시킵니다

    img = np.array(img).reshape(1,-1)
    # np.array 메서드를 사용하여 img를 배열 타입으로 변환합니다. reshape 메서드의 첫 번째 '1'은 이 2차원 배열을 1차원 배열로 확장하는 데 사용됩니다
    # 두 번째 '-1'은 Numpy가 나머지 차원의 크기를 자동으로 계산하여 784개의 요소를 포함하는 배열로 평탄화합니다

    predict = model.predict(img)
    # 처리된 img를 모델에 입력하여 예측하며, 모델은 설정된 가장 가까운 3개의 '이웃'에 따라 최종 결과를 반환합니다

    print(f'테스트 결과는: {predict[0]}')
    # 배열 형식으로 반환되므로 첫 번째 숫자를 출력하여 결과를 나타냅니다

if __name__ == '__main__': # 메인 함수
    digit_test()

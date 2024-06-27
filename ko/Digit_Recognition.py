import warnings  # 경고 메시지 무시
import sys  # 시스템 관련 기능
import joblib  # 훈련된 모델 로드 및 저장
import numpy as np  # 배열 작업
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel  # GUI 구성 모듈
from PyQt5.QtGui import QPixmap  # 이미지 표시 처리
from PIL import Image  # 이미지 파일 처리

class MainWindow(QWidget):
    # GUI 메인 창 클래스를 생성하고 QWidget의 메서드를 상속하여 창의 외관과 작동을 완전히 사용자 정의할 수 있음

    def __init__(self):
        super().__init__()
        # 이 초기화 함수를 실행할 때 부모 클래스(QWidget) 메서드를 호출

        self.init_ui()
        self.model = joblib.load('../mnist_784.pth')
        # joblib 메서드를 사용하여 훈련된 모델 로드

    def init_ui(self):
        self.setWindowTitle('손글씨 숫자 인식')
        # 이 창의 제목을 설정, 창이 열릴 때 상단 중앙에 표시

        self.resize(1000, 600)
        # 창 크기 조정

        layout = QVBoxLayout()
        # 수직 레이아웃을 생성, 이후 추가된 하위 구성 요소들이 위에서 아래로 수직으로 배열됨
        # 창 크기에 따라 하위 구성 요소의 크기와 위치가 실시간으로 조정되어 중첩되지 않고 균등하게 배치됨

        self.btn = QPushButton('이미지 로드', self)
        # 버튼을 생성하고 버튼에 "이미지 로드" 텍스트를 수평 중앙에 표시
        # 괄호 안의 self에 대한 간단한 설명: 이 self는 이 클래스를 버튼의 "부모 위젯"으로 지정
        # 버튼을 이 창에 추가, 창이 닫힐 때 버튼이 자동으로 소멸되어 메모리 누수를 방지

        self.btn.setFixedSize(200, 200)
        # 버튼 크기 조정

        self.btn.clicked.connect(self.load_Image)
        # 버튼의 클릭 신호를 self.loadImage 함수에 연결, 버튼이 클릭되면 이 함수가 트리거됨

        layout.addWidget(self.btn)
        # 버튼을 레이아웃에 추가

        self.resultLabel = QLabel('테스트 결과:', self)
        # 최종 결과를 표시할 레이블 생성

        layout.addWidget(self.resultLabel)
        # 결과 레이블을 레이아웃에 추가

        self.imageLabel = QLabel(self)
        # 테스트 이미지를 표시할 레이블 생성

        layout.addWidget(self.imageLabel)
        # 이미지 레이블을 레이아웃에 추가

        self.setLayout(layout)
        # 생성된 레이아웃을 현재 창의 레이아웃 관리자

        self.setLayout(layout)
        # 생성된 레이아웃을 현재 창의 레이아웃 관리자로 설정하여 레이아웃에 추가된 레이블이 자동으로 조정되고 표시됨

    def load_Image(self):
        options = QFileDialog.Options()
        # 이미지 클릭 시 트리거되는 파일 선택 상자 생성 메서드

        filename, _ = QFileDialog.getOpenFileName(self, "이미지를 선택하세요", "", "모든 파일 (*)", options=options)
        # 파일 선택 상자 열기 및 파일 선택을 대화 상자로 전달, ""는 기본 디렉터리, "모든 파일 (*)"은 모든 유형의 파일을 표시하고 선택할 수 있게 함

        if filename:
            pixmap = QPixmap(filename)
            # 선택한 이미지를 로드하기 위해 QPixmap 메서드 사용, QPixmap을 사용하는 주요 이유는 QLabel과 호환되며 imageLabel에 직접 로드할 수 있기 때문

            self.imageLabel.setPixmap(pixmap)
            # 로드된 이미지를 imageLabel로 설정하여 창에 표시

            self.imageLabel.adjustSize()
            # 이미지에 맞게 ImageLabel 크기 조정

            prediction = self.predict_Digit(filename)
            # 예측을 위해 predictDigit 함수를 호출하고 값을 prediction에 반환

            self.resultLabel.setText(f'테스트 결과: {prediction}')
            # 예측된 결과를 result_Label에 표시할 텍스트에 추가

    def predict_Digit(self, filename):
        img = Image.open(filename).convert('L')
        # Image.open 메서드를 사용하여 경로상의 이미지를 열고 convert 메서드를 사용하여 그레이스케일로 변환

        img = img.resize((28, 28))
        # 이미지 크기를 28*28 포맷으로 압축하여 모델의 입력 요구 사항을 충족

        img = np.array(img).reshape(1, -1)
        # np.array 메서드를 사용하여 img를 배열 타입으로 변환, reshape 메서드의 첫 번째 '1'은 이 2차원 배열을 1차원 배열로 확장하는 데 사용
        # 두 번째 '-1'은 Numpy가 나머지 차원의 크기를 자동으로 계산하여 배열을 784개의 요소를 포함하는 배열로 평탄화

        prediction = self.model.predict(img)
        # 처리된 img를 모델에 입력하여 예측, 모델은 설정된 가장 가까운 3개의 '이웃'에 기반하여 최종 결과를 반환
        return prediction[0]
        # 배열 형태로 반환되므로 그 중 첫 번째 숫자를 출력, 그 결과를 반환

if __name__ == '__main__':  # 메인 함수
    warnings.filterwarnings("ignore")
    # 훈련 시의 특징 이름과 현재 특징 이름이 달라서 경고가 발생하지만 실행 및 결과에는 영향을 미치지 않음
    # 따라서 이 경고를 무시하면 됨, 이를 해결하려면 모델을 테스트하기 전에 다음 줄을 코드에 추가
    # X.columns = [f'pixel{i}' for i in range(X.shape[1])]

    app = QApplication(sys.argv)
    # 이 프로그램의 제어 흐름 및 기타 설정을 관리하는 QApplication 객체 생성

    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
    # app.exec_()는 프로그램의 메인 루프에 들어가서 앞서 언급한 이벤트를 처리하기 시작함. 종료 시 프로그램이 깨끗하게 종료될 수 있도록 함

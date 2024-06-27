import warnings  # 警告メッセージを無視する
import sys  # システム関連の機能
import joblib  # 訓練済みモデルのロードと保存
import numpy as np  # 配列を操作する
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel  # GUI構築モジュール
from PyQt5.QtGui import QPixmap  # 画像表示を処理する
from PIL import Image  # 画像ファイルを処理する

class MainWindow(QWidget):
    # GUIメインウィンドウのクラスを作成し、QWidgetのメソッドを継承することで、ウィンドウの外観や操作を完全にカスタマイズできる

    def __init__(self):
        super().__init__()
        # この初期化関数を実行する際に、親クラス（QWidget）のメソッドを呼び出す

        self.init_ui()
        self.model = joblib.load('../mnist_784.pth')
        # joblibメソッドを使用して訓練済みモデルをロードする

    def init_ui(self):
        self.setWindowTitle('手書き数字認識')
        # このウィンドウのタイトルを設定する。ウィンドウが開いたとき、上部中央の枠に表示される

        self.resize(1000, 600)
        # ウィンドウのサイズを調整する

        layout = QVBoxLayout()
        # 垂直レイアウトを作成する。これにより、垂直レイアウトに追加されたサブウィジェットが上下に垂直に配置される
        # また、ウィンドウのサイズに応じてサブウィジェットのサイズと位置がリアルタイムで調整され、重ならないように均等に配置される

        self.btn = QPushButton('画像を読み込む', self)
        # ボタンを作成し、ボタン上に「画像を読み込む」というテキストを水平に中央揃えで表示する
        # 括弧内のselfの簡単な説明：このselfは、ボタンの「親ウィジェット」としてこのクラスを指定する
        # ボタンをこのウィンドウに追加し、ウィンドウが閉じられたときにボタンが自動的に破棄されるようにしてメモリリークを防ぐ

        self.btn.setFixedSize(200, 200)
        # ボタンのサイズを調整する

        self.btn.clicked.connect(self.load_Image)
        # このボタンのクリック信号をself.loadImage関数に接続し、ボタンがクリックされたときにこの関数がトリガーされるようにする

        layout.addWidget(self.btn)
        # このボタンをレイアウトに追加する

        self.resultLabel = QLabel('テスト結果は：', self)
        # 最終結果を表示するラベルを作成する

        layout.addWidget(self.resultLabel)
        # 結果ラベルをレイアウトに追加する

        self.imageLabel = QLabel(self)
        # テスト画像を表示するラベルを作成する

        layout.addWidget(self.imageLabel)
        # 画像ラベルをレイアウトに追加する

        self.setLayout(layout)
        # 作成したレイアウトを現在のウィンドウのレイアウトマネージャーとして設定し、レイアウトに追加されたラベルが自動的に調整されて表示されるようにする

    def load_Image(self):
        options = QFileDialog.Options()
        # 画像をクリックしたときにトリガーされるファイル選択ボックスを作成するメソッド

        filename, _ = QFileDialog.getOpenFileName(self, "画像を選択してください", "", "すべてのファイル (*)", options=options)
        # ファイル選択ボックスを開き、ファイルを選択してダイアログに渡す。""はデフォルトのディレクトリを表し、「すべてのファイル (*)」はすべての種類のファイルを表示して選択できるようにする

        if filename:
            pixmap = QPixmap(filename)
            # 選択された画像を読み込むためにQPixmapメソッドを使用する。QPixmapを使用する主な理由は、QLabelと互換性があり、imageLabelに直接読み込むことができるため

            self.imageLabel.setPixmap(pixmap)
            # 読み込んだ画像をimageLabelとして設定し、ウィンドウに表示できるようにする

            self.imageLabel.adjustSize()
            # 画像に合わせてImageLabelのサイズを調整する

            prediction = self.predict_Digit(filename)
            # 予測のためにpredictDigit関数を呼び出し、値をpredictionに返す

            self.resultLabel.setText(f'テスト結果は： {prediction}')
            # 予測された結果をresult_Labelに表示されるテキストに追加する

    def predict_Digit(self, filename):
        img = Image.open(filename).convert('L')
        # Image.openメソッドを使用してパス上の画像を開き、convertメソッドを使用してグレースケールに変換する

        img = img.resize((28, 28))
        # 画像のサイズを28*28のフォーマットに圧縮し、モデルの入力要件を満たす

        img = np.array(img).reshape(1, -1)
        # np.arrayメソッドを使用してimgを配列タイプに変換する。reshapeメソッドの最初の '1' は、この二次元配列を一次元配列に伸ばすために使用される
        # 二番目の '-1' はNumpyが残りの次元のサイズを自動的に計算できるようにし、配列を784個の要素を含む配列に平坦化する

        prediction = self.model.predict(img)
        # 処理されたimgをモデルに入力して予測し、モデルは設定された最も近い3つの '隣人' に基づいて最終結果を返す
        return prediction[0]
        # 配列の形式で返されるため、その中の最初の数値を出力し、それが結果であるため、その結果を返す

if __name__ == '__main__':  # メイン関数
    warnings.filterwarnings("ignore")
    # トレーニング時の特徴名と現在の特徴名が異なるため、警告が表示されるが、実行や結果には影響しない
    # したがって、この警告を無視するだけでよい。これを解決したい場合は、モデルをテストする前のコードに次の行を追加する
    # X.columns = [f'pixel{i}' for i in range(X.shape[1])]

    app = QApplication(sys.argv)
    # このプログラムの制御フローと他の設定を管理するQApplicationオブジェクトを作成する

    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
    # app.exec_()はプログラムのメインループに入り、上記のイベントの処理を開始する。終了時にプログラムがクリーンに終了できるようにする

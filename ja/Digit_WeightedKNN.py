import warnings   # 警告メッセージを無視する
import joblib   # 訓練済みモデルのロードと保存
import numpy as np  # 配列を操作する
from PIL import Image  # 画像ファイルを処理する

class DigitRecognizer:
    # 手書き数字認識器のクラスを作成する

    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        # joblibメソッドを使用して訓練済みモデルをロードする
        self.X_model = self.model._fit_X
        self.y_model = self.model._y
        # モデルからトレーニングデータとラベルを抽出する

    def compute_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
        # 式を使用して2つの点間のユークリッド距離を計算する

    def compute_weight(self, distance):
        a, b = 1, 1
        # 重みを調整するためのパラメータを定義する。aはゼロ除算を回避するためのスムーサーと理解できる
        return b / (distance + a)
        # 重みの値を計算して返す。この式では、距離が小さいほど重みが大きく、距離が大きいほど重みが小さくなることがわかる

    def predict_digit(self, filename):
        img = Image.open(filename).convert('L')
        img = img.resize((28, 28))
        img = np.array(img).reshape(1, -1)
        # 入力された画像を処理する。これについては以前説明したので、繰り返しません

        distances = []
        # 入力画像と各トレーニングサンプルの距離とラベルを格納するための空のリストを作成する
        for i, X_train in enumerate(self.X_model):
            # トレーニングセットの各サンプルを繰り返し処理する

            distance = self.compute_distance(img, X_train.reshape(1, -1))
            # compute_distance関数を使用して画像とトレーニングサンプル間の距離を計算する。トレーニングサンプルをimgの形状に一致するようにリシェイプすることを忘れないでください

            weight = self.compute_weight(distance)
            # 重みを計算する

            distances.append((weight, self.y_model[i]))
            # 重みと対応するラベルをタプルとしてリストに追加する

        distances.sort(key=lambda x: x[0], reverse=True)
        # ラムダ式を使用してリストを重みの降順でソートする

        k_neighbors = distances[:3]
        # ソート後、重みが最も高い3つの隣接点を選択する

        weighted_votes = {}
        # 各ラベルの重み付き投票結果を記録するための空の辞書を作成する

        for weight, label in k_neighbors:
            # 3つの隣接点の重みとラベルを繰り返し処理する

            if label in weighted_votes:
                weighted_votes[label] += weight
                # ラベルが辞書に既に存在する場合、この重みを累積するだけでよい

            else:
                weighted_votes[label] = weight
                # そうでない場合、現在の重みでこのラベルの新しいエントリを作成する

        predictions = max(weighted_votes, key=weighted_votes.get)
        # 重み付き投票結果で最も重みが高いラベルを最終予測結果として選択する

        return predictions
        # この結果を返す

def digit_test():
    warnings.filterwarnings("ignore")
    # トレーニング時の特徴名と現在の特徴名が異なるため、警告が発生しますが、実行や結果には影響しません
    # したがって、この警告を無視してください。解決したい場合は、モデルをテストする前に次のコード行を追加してください
    # X.columns = [f'pixel{i}' for i in range(X.shape[1])]

    recognizer = DigitRecognizer('../mnist_784.pth')
    filename = '../handwritten_digits/digit_1.png'
    prediction = recognizer.predict_digit(filename)
    print(f'テスト結果は: {prediction}')
    # 予測画像については、以前のファイルで説明しましたので、繰り返しません

if __name__ == '__main__':  # メイン関数
    digit_test()

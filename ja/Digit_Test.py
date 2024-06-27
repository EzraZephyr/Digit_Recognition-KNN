import warnings   # 警告メッセージを無視する
import joblib   # 訓練済みモデルのロードと保存
import numpy as np  # 配列を操作する
from PIL import Image  # 画像ファイルを処理する

def digit_test():
    warnings.filterwarnings("ignore")
    # トレーニング時の特徴名と現在の特徴名が異なるため、警告が発生しますが、実行や結果には影響しません
    # したがって、この警告を無視してください。解決したい場合は、モデルをテストする前に次のコード行を追加してください
    # X.columns = [f'pixel{i}' for i in range(X.shape[1])]

    model = joblib.load('../mnist_784.pth')
    # joblibメソッドを使用して訓練済みモデルをロードする

    filename = '../handwritten_digits/digit_1.png'
    # 画像ファイルのパスをfilenameに保存し、以下で直接呼び出せるようにする

    img = Image.open(filename).convert('L')
    # Image.openメソッドを使用してそのパスの画像を開き、convertメソッドを使用してグレースケールに変換する

    img = img.resize((28, 28))
    # 画像のサイズを28*28のフォーマットに圧縮し、モデルの入力要件を満たす

    img = np.array(img).reshape(1,-1)
    # np.arrayメソッドを使用してimgを配列タイプに変換する。reshapeメソッドの最初の '1' は、この2次元配列を1次元配列に伸ばすために使用される
    # 2番目の '-1' はNumpyが残りの次元のサイズを自動的に計算し、784個の要素を含む配列に平坦化する

    predict = model.predict(img)
    # 処理されたimgをモデルに入力して予測し、モデルは設定された最も近い3つの '隣人' に基づいて最終結果を返す

    print(f'テスト結果は: {predict[0]}')
    # 配列形式で返されるため、その中の最初の数値を出力し、それが結果となる

if __name__ == '__main__': # メイン関数
    digit_test()

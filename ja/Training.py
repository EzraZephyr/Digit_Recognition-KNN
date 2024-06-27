import pandas as pd    # データセットを処理する
import joblib   # 訓練済みモデルのロードと保存
from sklearn.datasets import fetch_openml   # 手書き数字画像
from sklearn.model_selection import train_test_split    # 訓練データとテストデータを分割し、比率を調整する
from sklearn.neighbors import KNeighborsClassifier  # K-最近傍分類器、つまりKNNアルゴリズム。最も近いK個の隣接点に基づいて予測する
from sklearn.metrics import accuracy_score  # モデルの精度を計算する

mnist = fetch_openml('mnist_784', version=1)
# mnist_784という名前のデータセットを取得する。バージョンは1。このデータセットには0〜9の手書き数字の画像が70,000枚含まれている
# 興味深い背景、これらのデータセットのうち60,000枚の訓練画像はアメリカの国勢調査局の職員によって書かれたもので、10,000枚のテスト画像はアメリカの高校生によるものである

X = pd.DataFrame(mnist['data'])
# このデータセットから特徴を抽出する。これらの特徴はデータセット内で 'data' というインデックスを持つ二次元ベクトルなので、DataFrameを使って処理する必要がある
y = pd.Series(mnist.target).astype('int')
# このデータセット内のラベル、すなわち28*28の二次元ベクトルに対応する目標値は一次元ベクトルなので、Seriesで処理する
# 目標値は文字列タイプとして設計されているので、astype('int') を使ってintタイプに変換する必要がある

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# train_test_split関数を使って特徴とラベル（つまりXとy）を80%と20%の割合で分割する
# これによりX_trainとy_trainは訓練に使用され、X_testとy_testはテストに使用される

estimator = KNeighborsClassifier(n_neighbors=3)
# K-最近傍分類器をインスタンス化し、n_neighborsを3に設定する。これは将来のデータテストやモデル使用時に
# このアルゴリズムが特徴空間内で最も近い3つの隣接点（一般的にはユークリッド距離を使用）を自動的に検索し
# これらの3つの隣接点のカテゴリに基づいて新しいデータポイントのカテゴリを決定することを意味する

estimator.fit(X_train, y_train)
# この分類器のfitメソッドを使用してモデルの訓練を開始し、訓練データとその対応するラベルを渡す
# モデルがデータの特徴と目標値の関係を学習し、記憶することで、将来的に予測に使用できるようにする

y_pred = estimator.predict(X_test)
# モデルの訓練が完了した後、上記で分割したテストデータを使用して予測を行い、モデルの精度を計算する
# 分類器のpredictメソッドを呼び出し、テストデータX_testを渡すと、学習したルールに基づいて各テストサンプルの予測結果を返す

print(accuracy_score(y_test, y_pred))
# テスト結果を比較してモデルの精度を計算する

joblib.dump(estimator, '../mnist_784.pth')
# 最後にjoblibライブラリのdumpメソッドを呼び出し、訓練済みモデルを渡してファイル名を 'mnist_784.pth' に設定する
# これにより、モデルがディスクに保存され、再訓練することなく直接ロードして使用できるようになる

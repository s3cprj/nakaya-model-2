from PIL import Image
import os, glob
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# 画像分類のためのラベル（クラス名）を設定
classes = ["open","close"]
# クラス数を取得
num_classes = len(classes)
# 画像をリサイズする際のサイズを設定
image_size = 128

# 画像データセットが格納されているディレクトリのパスを設定
datadir='./'

# 画像データ（特徴量）とラベルデータを格納するリストを初期化
X = []
Y = []

# 各クラスの画像データを読み込み、前処理を行う
for index, classlabel in enumerate(classes):
    # 各クラスの画像が格納されているディレクトリのパスを生成
    photos_dir = datadir + classlabel
    # 対象のディレクトリ内の全画像ファイル（jpg）のパスを取得
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        # 画像ファイルを開き、RGB形式に変換後、指定されたサイズにリサイズ
        image = Image.open(file).convert("RGB").resize((image_size, image_size))
        # リサイズされた画像データをNumPy配列に変換
        data = np.asarray(image)

        # 画像を特定の角度で回転させ、また左右反転させたデータを追加
        for angle in range(-20, 20, 5):
            # 画像を指定した角度で回転させ、データセットに追加
            img_r = image.rotate(angle)
            data = np.asarray(img_r)
            X.append(data)
            Y.append(index)

            # 画像を左右反転させ、データセットに追加
            img_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
            data = np.asarray(img_trans)
            X.append(data)
            Y.append(index)

# 画像データとラベルデータをNumPy配列に変換
X = np.array(X)
Y = np.array(Y)

# データセットを訓練データとテストデータに分割（テストデータの割合は20%）。
(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2)

# 画像データを正規化
X_train = X_train.astype("float") / 255
X_test = X_test.astype("float") / 255

# ラベルデータをOne-Hotエンコーディング形式に変換
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# 前処理済みのデータセットをファイルに保存
xy = (X_train, X_test, y_train, y_test)
np.save("./dataset.npy", xy)

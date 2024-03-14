import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

import glob
from PIL import Image
import matplotlib.pyplot as plt

import os
from keras.callbacks import TensorBoard,ModelCheckpoint

import pickle



# ハイパーパラメーターの設定
hp1 = {}
hp1['class_num'] = 2 # 分類するクラスの数(open or close)
hp1['batch_size'] = 32 # バッチサイズ(一度に処理する画像の数)
hp1['epoch'] = 5 # エポック数(訓練の繰り返し回数)

# .pklファイルのパス
file_path = './dataset.pkl'

# .pklファイルを読み込む
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# データセットの読み込み
X_train, X_test, y_train, y_test = data


# 入力データの形状を設定
input_shape=X_train.shape[1:]

# CNNを構築
def CNN(input_shape):
        model = Sequential()

        # 入力層に対して32フィルタの畳み込み層を追加
        model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
        # 活性化関数であるReLU関数を使用
        model.add(Activation('relu'))
        # 32フィルタの畳み込み層を追加
        model.add(Conv2D(32, (3, 3)))
        # バッチ正規化
        model.add(BatchNormalization())
        # 活性化関数
        model.add(Activation('relu'))
        # プーリング層を追加
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # ドロップアウトで過学習を防止
        model.add(Dropout(0.25))

        # 64フィルタの畳み込み層を追加
        model.add(Conv2D(64, (3, 3), padding='same'))
        # 活性化関数
        model.add(Activation('relu'))
        # 64フィルタの畳み込み層を追加
        model.add(Conv2D(64, (3, 3)))
        # バッチ正規化
        model.add(BatchNormalization())
        # 活性化関数
        model.add(Activation('relu'))
        # プーリング層を追加
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # ドロップアウトで過学習を防止
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))


        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(hp1['class_num']))
        model.add(Activation('softmax'))

        return model


# モデルを選択
model=CNN(input_shape)

# モデルのコンパイル
# 損失関数、最適化アルゴリズム、評価指標を設定
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

# 訓練時のコールバック設定
log_dir = os.path.join(os.path.dirname(__file__), "logdir")
model_file_name="model_file.hdf5"

# モデルの訓練
# 訓練データと検証データに分け、エポック数だけ訓練を繰り返す
# TensorBoardとModelCheckpointをコールバックとして使用
history = model.fit(
        X_train, y_train,
         epochs=hp1['epoch'],
         validation_split = 0.2,
         callbacks=[
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(os.path.join(log_dir,model_file_name),save_best_only=True)
                ]
        )

# モデルの評価
# テストデータセットを使用して損失と精度を計算
loss,accuracy = model.evaluate(X_test, y_test, batch_size=hp1['batch_size'])

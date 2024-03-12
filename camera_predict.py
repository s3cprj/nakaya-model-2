import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# 学習済みモデルの読み込み
model = load_model("logdir/model_file.hdf5")

# カメラの設定
cap = cv2.VideoCapture(0)

while True:
    # フレームを読み込む
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    # OpenCVのフレームはBGR形式なので、RGBに変換
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # PIL.Imageオブジェクトに変換
    image_pil = Image.fromarray(frame_rgb)
    # モデルの入力に合わせてリサイズ
    image_resized = image_pil.resize((128, 128))
    # NumPy配列に変換
    image_np = np.asarray(image_resized)
    # 正規化
    image_np = image_np.astype('float32') / 255.0
    # バッチ次元を追加
    image_np_expanded = np.expand_dims(image_np, axis=0)
    
    # 予測を行う
    predictions = model.predict(image_np_expanded)
    class_index = np.argmax(predictions, axis=1)
    
    # 予測結果に基づいて何かを行う
    if class_index == 0:
        print("Class 'open' detected")
    else:
        print("Class 'close' detected")
    
    # リアルタイム映像を表示
    cv2.imshow('frame', frame)
    
    # qを押して終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラを解放
cap.release()
cv2.destroyAllWindows()

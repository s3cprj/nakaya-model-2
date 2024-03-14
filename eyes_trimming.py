import cv2
import os

def detect_and_crop_eyes_from_directory(image_directory, output_folder):
    # Haar Cascadeファイルのパス
    face_cascade_path = 'haarcascade_frontalface_default.xml'
    eye_cascade_path = 'haarcascade_eye.xml'
    
    # カスケード分類器をロード
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + eye_cascade_path)
    
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 指定されたディレクトリ内のすべての画像ファイルに対して処理を実行
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_directory, filename)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 顔を検出
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                
                # 目を検出
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                    
                    # トリミングされた目の画像を保存
                    eye_img_path = os.path.join(output_folder, f"{filename}_eye_{ex}_{ey}.jpg")
                    cv2.imwrite(eye_img_path, eye_img)

# 使用例
image_directory = 'datasets/close'  # 画像ファイルがあるディレクトリへのパス
output_folder = 'datasets/close_eyes'  # 出力フォルダへのパス

detect_and_crop_eyes_from_directory(image_directory, output_folder)

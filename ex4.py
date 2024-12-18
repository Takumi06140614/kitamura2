import cv2
import torch
from ultralytics import YOLO
import math

# 画像をimgに読み込み
# img = cv2.imread("ファイル名")
img = cv2.imread("ex1.jpg")

video_path = "https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex3b.mp4"

# 動画を開く
cap = cv2.VideoCapture(video_path)

model = YOLO("yolov8x-pose.pt")

results = model("https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex1.jpg", save=True, save_txt=True, save_conf=True)
keypoints = results[0].keypoints
keypoints_data = keypoints.data

# Tensorの形のデータをListの形に変換する
keypoints_data = keypoints_data.tolist()

# 新たなListの作成
new_data1 = []

for i in range(len(keypoints_data[0])):
    new_data1.append(keypoints_data[0][i][:2])  # 0番目と1番目の要素だけを取得

# フレーム番号を0にする
cnt = 0

# フレームを読み出す
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # フレーム番号をフレームに描画
        cv2.putText(frame, str(cnt), (100, 300), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)

        # YOLOの設定
        results = model(frame)
        keypoints = results[0].keypoints
        keypoints_data = keypoints.data
        
        # Tensorからリストに変換
        keypoints_data = keypoints_data.tolist()

        # 新しいリストの作成
        new_data2 = []
        for i in range(len(keypoints_data[0])):
            new_data2.append(keypoints_data[0][i][:2])
        
        count = 0
        
        for i in range(len(keypoints_data[0])):
            if new_data2[i][0]-new_data1[i][0] <= 0.1:
                count = count + 1
            if new_data2[i][1]-new_data1[i][1] <= 0.1:
                count = count + 1
        
        cnt_final = 0
        
        if  count == 24:
            cnt_final = cnt
            break
        
        # フレーム番号をインクリメント
        cnt += 1

        # ESCキーで終了
        if cv2.waitKey(20) == 27:
            break
    else:
        break

print(cnt)

# リソースの解放
cap.release()
cv2.destroyAllWindows()
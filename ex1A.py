import cv2
import torch
from ultralytics import YOLO

# 画像をimgに読み込み
# img = cv2.imread("ファイル名")
img = cv2.imread("ex1.jpg")

model = YOLO("yolov8x-pose.pt")

results = model("https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex1.jpg", save=True, save_txt=True, save_conf=True)
keypoints = results[0].keypoints
keypoints_data = keypoints.data

# Tensorの形のデータをListの形に変換する
keypoints_data = keypoints_data.tolist()


# 新たなListの作成
new_data = []

for i in range(len(keypoints_data[0])):
    new_data.append(keypoints_data[0][i][:2])  # 0番目と1番目の要素だけを取得

   
# 円の描画(画像,中心座標,半径,色,線の太さ,線の種類)(5,6,11,12)
x_data = (int(keypoints_data[0][5][0])+int(keypoints_data[0][6][0])+int(keypoints_data[0][11][0])+int(keypoints_data[0][12][0]))/4
y_data = (int(keypoints_data[0][5][1])+int(keypoints_data[0][6][1])+int(keypoints_data[0][11][1])+int(keypoints_data[0][12][1]))/4
cv2.circle(img,(int(x_data),int(y_data)),5,(0,255,255),-1,cv2.LINE_8)

cv2.imwrite('ex1A.1.jpg',img )

#imgの中身を表示する
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()



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

# 線の描画(画像,端の座標,端の座標,色,太さ,種類)
# 5-7,7-9,5-6,6-8,6-10,6-12,5-11,11-12,12-14,14-16,11-13,13-15
line_list = [5,7,7,9,5,6,6,8,6,10,6,12,5,11,11,12,12,14,14,16,11,13,13,15]
x = 0 
for i in range(5, len(new_data)):
    start_point = (int(new_data[line_list[x]][0]), int(new_data[line_list[x]][1]))
    end_point = (int(new_data[line_list[x+1]][0]), int(new_data[line_list[x+1]][1]))
    cv2.line(img, start_point, end_point, (0, 0, 255), 3, cv2.LINE_8)
    x += 2
    
    
# 円の描画(画像,中心座標,半径,色,線の太さ,線の種類)
for i in range(5,len(new_data)):
   cv2.circle(img,(int(new_data[i][0]),int(new_data[i][1])),5,(0,255,255),-1,cv2.LINE_8)

cv2.imwrite('ex_draw.jpg',img )

#imgの中身を表示する
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()



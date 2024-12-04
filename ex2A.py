from ultralytics import YOLO
import cv2

# 画像の読み込み
img = cv2.imread("ex2.jpg")

model = YOLO("yolov8x.pt")

results = model("https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex2.jpg", save=True, save_txt=True, save_conf=True)
boxes = results[0].boxes
boxes_data = boxes.data

# Tensorの形のデータをListの形に変換する
boxes_data = boxes_data.tolist()

# 新たなリストの作成
new_data = []
for i in range(len(boxes_data)):
    new_data.append(boxes_data[i][:4])  # 4番目までの要素を取得(始点x,始点y,終点x,終点y)

# 最大の面積のものを求める
square_A = 0
A_i = 0
for i in range(len(boxes_data)):
    square_B = (new_data[i][2]-new_data[i][0]) * (new_data[i][3]-new_data[i][1])
    
    if square_B > square_A:
      square_A = square_B
      A_i = i


    
# 長方形の描画
start_point = (int(new_data[A_i][0]), int(new_data[A_i][1]))
end_point = (int(new_data[A_i][2]), int(new_data[A_i][3]))
cv2.rectangle(img, start_point, end_point, (0, 0, 255), thickness=4, lineType=cv2.LINE_4)

# 画像の書き出し
cv2.imwrite('ex2A.1.jpg',img )


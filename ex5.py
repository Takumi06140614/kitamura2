from ultralytics import YOLO
import cv2
from PIL import Image

# 画像の読み込み
img = cv2.imread("ex2.jpg")
Img = Image.open("ex2.jpg")

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

for i in range(len(new_data)):
    subtraction_x = int(new_data[i][2]) - int(new_data[i][0])
    subtraction_y = int(new_data[i][3]) - int(new_data[i][1])
    
    total = 0
    
    for j in range(subtraction_x):
        for k in range(subtraction_y):
            
            r,g,b = Img.getpixel((int(new_data[i][0]) + j, int(new_data[i][1] + k)))
            
            if b > 230:
                total += 1
    
    if total > 30:
        start_point = (int(new_data[i][0]), int(new_data[i][1]))
        end_point = (int(new_data[i][2]), int(new_data[i][3]))
        cv2.rectangle(img, start_point, end_point, (0, 0, 255), thickness=4, lineType=cv2.LINE_4)
        
cv2.imwrite('ex5.jpg',img )
    
    
    


    
# 長方形の描画
# for i in range(len(boxes_data)):
#     start_point = (int(new_data[i][0]), int(new_data[i][1]))
#     end_point = (int(new_data[i][2]), int(new_data[i][3]))
#     cv2.rectangle(img, start_point, end_point, (0, 0, 255), thickness=4, lineType=cv2.LINE_4)

# # 画像の書き出し
# cv2.imwrite('ex2.1.jpg',img )
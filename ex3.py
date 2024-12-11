import cv2
import torch
from ultralytics import YOLO
import math

# 角度を求める
def calculate_angle(ax, ay, bx, by, cx, cy):
    
    # ベクトルBAとBCを計算
    vec_ba = (ax - bx, ay - by)
    vec_bc = (cx - bx, cy - by)

    # 内積を計算
    dot_product = vec_ba[0] * vec_bc[0] + vec_ba[1] * vec_bc[1]

    # ベクトルの大きさを計算
    magnitude_ba = math.sqrt(vec_ba[0]**2 + vec_ba[1]**2)
    magnitude_bc = math.sqrt(vec_bc[0]**2 + vec_bc[1]**2)

    # コサイン値を計算
    cos_theta = dot_product / (magnitude_ba * magnitude_bc)

    # # コサイン値の範囲をクリップ
    # cos_theta = max(-1.0, min(1.0, cos_theta))

    # 角度をラジアンで計算し、度に変換
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

video_path = "https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex3b.mp4"

# YOLOの設定
model = YOLO("yolov8x-pose.pt")

# 動画を開く
cap = cv2.VideoCapture(video_path)

# 各種プロパティの設定
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 動画に書き込み設定
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

# フレーム番号を0にする
cnt = 0

angles = []
# フレームを読み出す
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # フレーム番号をフレームに描画
        # cv2.putText(frame, str(cnt), (100, 300), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)

        # YOLOの設定
        results = model(frame)
        keypoints = results[0].keypoints
        keypoints_data = keypoints.data
        
        # Tensorからリストに変換
        keypoints_data = keypoints_data.tolist()

        # 新しいリストの作成
        new_data = []
        for i in range(len(keypoints_data[0])):
            new_data.append(keypoints_data[0][i][:2])

        # 線の描画 (5-7,7-9,5-6,6-8,8-10など)
        line_list = [5,7,7,9,5,6,6,8,8,10,6,12,5,11,11,12,12,14,14,16,11,13,13,15]
        x = 0
        for i in range(5, len(new_data)):
            start_point = (int(new_data[line_list[x]][0]), int(new_data[line_list[x]][1]))
            end_point = (int(new_data[line_list[x+1]][0]), int(new_data[line_list[x+1]][1]))
            cv2.line(frame, start_point, end_point, (255, 0, 0), 3, cv2.LINE_8)
            x += 2

        # 6-8,8-10の角度を求めて、80°～100°の場合に赤色にする
        angle = calculate_angle(new_data[8][0], new_data[8][1], new_data[6][0], new_data[6][1], new_data[12][0], new_data[12][1])
        angles.append(angle)
        if 80 <= angle <= 100:
            # 赤色に変更
            start_point = (int(new_data[6][0]), int(new_data[6][1]))
            end_point = (int(new_data[8][0]), int(new_data[8][1]))
            cv2.line(frame, start_point, end_point, (0, 0, 255), 3, cv2.LINE_8)
            start_point = (int(new_data[8][0]), int(new_data[8][1]))
            end_point = (int(new_data[10][0]), int(new_data[10][1]))
            cv2.line(frame, start_point, end_point, (0, 0, 255), 3, cv2.LINE_8)
        cv2.putText(frame, str(cnt), (100, 300), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
        # フレーム番号をインクリメント
        cnt += 1

        # フレームを書き込む
        out.write(frame)

        # # フレームを表示
        cv2.imshow("Frame", frame)

        # ESCキーで終了
        if cv2.waitKey(20) == 27:
            break
    else:
        break

# print(angles)
matches = [i for i in range(len(angles)) if 80 <= angles[i] <= 100]
print(matches)
# リソースの解放
cap.release()
out.release()
cv2.destroyAllWindows()
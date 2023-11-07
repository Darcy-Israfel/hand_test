import os
import pickle
import socket
import struct
from time import sleep

import mediapipe as mp
import cv2
import numpy as np


def get_str_gesture(out_fingers):
    if len(out_fingers) == 1 and out_fingers[0] == 4:
        str_gesture = 'Good'
    elif len(out_fingers) == 1 and out_fingers[0] == 20:
        str_gesture = '0'
    elif len(out_fingers) == 1 and out_fingers[0] == 8:
        str_gesture = '1'
    elif len(out_fingers) == 2 and out_fingers[0] == 8 and out_fingers[1] == 12:
        str_gesture = '2'
    elif len(out_fingers) == 2 and out_fingers[0] == 4 and out_fingers[1] == 20:
        str_gesture = '6'
    elif len(out_fingers) == 2 and out_fingers[0] == 4 and out_fingers[1] == 8:
        str_gesture = '8'
    elif len(out_fingers) == 3 and out_fingers[0] == 8 and out_fingers[1] == 12 and out_fingers[2] == 16:
        str_gesture = '3'
    elif len(out_fingers) == 3 and out_fingers[0] == 4 and out_fingers[1] == 8 and out_fingers[2] == 12:
        str_gesture = '7'
    elif len(out_fingers) == 4 and out_fingers[0] == 8 and out_fingers[1] == 12 and out_fingers[2] == 16 and \
            out_fingers[3] == 20:
        str_gesture = '4'
    elif len(out_fingers) == 5:
        str_gesture = '5'
    elif len(out_fingers) == 0:
        str_gesture = '0'
    else:
        str_gesture = ''
    return str_gesture


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


def run():
    while True:
        # 读取一帧图像
        ret, img = cap.read()
        img = cv2.flip(img, 1)  # 如果左右手相反的话需要用到图像水平翻转
        height, width, channels = img.shape
        # 转换为RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 得到检测结果
        results = hands.process(imgRGB)
        str_gesture = "null"
        strHandType = "null"

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
            # 获取左右手
            handType = results.multi_handedness
            strHandType = handType[0].classification[0].label
            # 采集所有关键点坐标
            list_lms = []
            for i in range(21):
                pos_x = int(hand.landmark[i].x * width)
                pos_y = int(hand.landmark[i].y * height)
                list_lms.append([pos_x, pos_y])

            # 构造凸包点
            list_lms = np.array(list_lms, dtype=np.int32)
            hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17]
            hull = cv2.convexHull(list_lms[hull_index], True)
            cv2.polylines(img, [hull], True, (0, 255, 0), 2)  # 凸包范围

            # 查找外部的点数
            ll = [4, 8, 12, 16, 20]
            out_fingers = []
            for i in ll:
                pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                dist = cv2.pointPolygonTest(hull, pt, True)
                if dist < 0:
                    out_fingers.append(i)

            str_gesture = get_str_gesture(out_fingers)
            cv2.putText(img, strHandType, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4, cv2.LINE_AA)
            cv2.putText(img, str_gesture, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4, cv2.LINE_AA)
            for i in ll:
                pos_x = int(hand.landmark[i].x * width)
                pos_y = int(hand.landmark[i].y * height)
                cv2.circle(img, (pos_x, pos_y), 3, (0, 255, 255), -1)  # 绘制指头尖的黄色


        cv2.imshow('hands', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()


if __name__ == '__main__':


    run()


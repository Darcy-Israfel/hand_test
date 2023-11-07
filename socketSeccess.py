import os
import pickle
import socket
import struct
import threading
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



receive = "no receive"
def handle_client(client_socket):
    """处理客户端请求的函数"""
    global receive
    while True:
        # 接收客户端发送的数据
        receive = client_socket.recv(10)
        receive = receive.decode()



def run():
    global receive
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
        #image_path = r"C:\Users\Darcy\Desktop\img.png"
        #cv2.imwrite(image_path, img)
        #
        # with open(r"C:\Users\Darcy\Desktop\file.txt", "w") as file:
        #     file.write(str_gesture)
        # 发送结果
        client_socket.sendall("result".encode())
        sendstr = strHandType + ":" + str_gesture
        client_socket.sendall(sendstr.encode())
        print(sendstr)
        # 发送图片
        sleep(0.0001) # 不然会收到picture
        client_socket.sendall("picture".encode())
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, img_encode = cv2.imencode('.jpg', img, encode_param)
        picture = np.array(img_encode)
        str_picture = picture.tostring()

        # 发送
        client_socket.send(str_picture)

        print(receive)
        if receive == "stop":
            break

        #cv2.imshow('hands', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()


if __name__ == '__main__':
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 12345))
    server_socket.listen(1)


    print("等待客户端连接...")
    client_socket, client_address = server_socket.accept()
    print(f"客户端 {client_address} 已连接")
    # 创建一个线程，监听客户端返回的值
    t = threading.Thread(target=handle_client, args=(client_socket,))
    t.start()
    run()


    client_socket.close()
    server_socket.close()



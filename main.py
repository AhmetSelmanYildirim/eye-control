import time
from math import hypot
import cv2
import dlib
import numpy as np
import vlc

# Alerts
eyes_closed_alert = vlc.MediaPlayer("eyes_closed_alert.mp3")
eyes_not_detected_alert = vlc.MediaPlayer("eyes_not_detected_alert.mp3")
blinked_minimum_times_alert = vlc.MediaPlayer("blinked_minimum_times_alert.mp3")
# Times
blink_time_start = time.time()
blink_time_finish = time.time()
time_control = time.time()
blink_counter_control_time = time.time()
blink_counter_control_time_test = time.time()
# Const
closed_blink_ratio = 4
eyes_close_time = 4
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#Variables
blink_counter_control = 0
blink_counter = 0


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio


def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye)
    right_eye = np.array(right_eye)
    return left_eye, right_eye

while True:
    _, frame = cap.read()   # Her framei okur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Datanin daha hizli islenebilmesi ve cpuyu yormamasi icin griye cevrildi

    faces = detector(gray)  # Yuzu algilar ve framedeki kapladigi yerin koordinatlarini tutar

    timestamp = time.time()
    cv2.putText(frame, "Blink counter: {}".format(blink_counter), (250, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

    for face in faces:
        eyes_not_detected_alert.stop()
        time_control = timestamp    # Gozun tespit edildigi durum icin

        landmarks = predictor(gray, face)
        left_eye, right_eye = eyes_contour_points(landmarks)

        # Goz kirpma orani tanimlama
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Gozler acikken yesil ile ciz
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        blink_time_finish = time.time()

        if blinking_ratio > closed_blink_ratio:       # Goz kapaliysa

            # Gozler kapandiginde kirmizi ile ciz
            cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
            cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)
            cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, "Time: {:.2f}".format(blink_time_finish - blink_time_start), (480, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

            blink_counter_control += 1

            if blink_time_finish - blink_time_start > eyes_close_time:
                eyes_closed_alert.play()

        elif blinking_ratio < closed_blink_ratio:     # Goz aciksa
            blink_time_start = blink_time_finish
            eyes_closed_alert.stop()
            cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

        # print(f'left eye : {left_eye}')                 #left eye : [[192 209] [195 203] [201 203] [208 208] [202 210] [196 211]]
        # print(f'right eye : {right_eye}')               #right eye : [[231 209] [237 203] [246 204] [255 208] [246 211] [237 211]]
        # print(f'left_eye_ratio : {left_eye_ratio}')     #left_eye_ratio : 2.2671568097509267
        # print(f'right_eye_ratio : {right_eye_ratio}')   #right_eye_ratio : 3.0026030373660784
        # print(f'blinking_ratio : {blinking_ratio}')     #blinking_ratio : 2.6348799235585023

    # Gozler tespit edilemiyorsa uyar
    time_fark = timestamp - time_control
    if time_fark > 2:
        cv2.putText(frame, "Eyes cannot be detected", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        eyes_not_detected_alert.play()

    # dakikada goz kapama sayisi
    if timestamp - blink_counter_control_time_test > 4:
        #print(f"blink_counter_control_time_test: {timestamp - blink_counter_control_time_test}")
        blinked_minimum_times_alert.stop()

        if blink_counter_control > 0:
            blink_counter += 1
            blink_counter_control = 0
            #print(f"goz kapama sayisi: {blink_counter}")


            if blink_counter < 10 and timestamp - blink_counter_control_time > 60:
                blinked_minimum_times_alert.play()

                #print(f"blink_counter_control_time: {timestamp - blink_counter_control_time}")
                blink_counter_control_time = timestamp
                blink_counter = 0

        blink_counter_control_time_test = timestamp

    cv2.imshow("Goz Takip Uygulamasi", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
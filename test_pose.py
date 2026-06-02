import cv2
import mediapipe as mp
import math
import numpy as np
import time

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

active_screen_time = 0
last_time = time.time()

while True:
    success, frame = cap.read()
    if not success: break

    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)

    if results.pose_landmarks:
        active_screen_time += dt
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        try:

            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x * w, 
                        landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y * h]
            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x * w, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y * h]

            ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w, 
                   landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, 
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, 
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]

            current_eye_dist = math.dist(left_eye, right_eye)
            current_angle = calculate_angle(ear, shoulder, hip)

            if current_eye_dist > 80: 
                status = "WARNING: TOO CLOSE!"
                color = (0, 165, 255) # สีส้ม

            elif current_angle < 145:
                status = "WARNING: SLOUCHING!"
                color = (0, 0, 255) # สีแดง
 
            else:
                status = "GOOD POSTURE"
                color = (0, 255, 0) # สีเขียว
            

            cv2.putText(frame, status, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            cv2.putText(frame, f"Eye: {int(current_eye_dist)}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Angle: {int(current_angle)}", (w - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        except Exception:
            pass
    else:
        cv2.putText(frame, "NO PERSON DETECTED", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 3)

    mins, secs = divmod(int(active_screen_time), 60)
    cv2.rectangle(frame, (0, h - 40), (250, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Time: {mins:02d}:{secs:02d}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Office Syndrome AI Tracker", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
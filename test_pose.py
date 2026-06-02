import cv2
import mediapipe as mp
import math
import numpy as np
import time

def get_vertical_angle(p1, p2):
    dx = p1[0] - p2[0]
    dy = p2[1] - p1[1]
    if dy == 0: return 90.0
    return math.degrees(math.atan2(abs(dx), abs(dy)))

def get_ear(eye_points):
    v1 = math.dist(eye_points[1], eye_points[5])
    v2 = math.dist(eye_points[2], eye_points[4])
    h1 = math.dist(eye_points[0], eye_points[3])
    if h1 == 0: return 0.0
    return (v1 + v2) / (2.0 * h1)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
active_screen_time = 0
last_time = time.time()

EAR_THRESHOLD = 0.15 

while True:
    success, frame = cap.read()
    if not success: break

    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb)
    face_results = face_mesh.process(rgb)

    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)

    is_looking = True
    is_eyes_open = True
    avg_ear = 0.0

    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark
        

        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        left_eye = [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in left_eye_indices]
        right_eye = [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in right_eye_indices]
        
        left_ear = get_ear(left_eye)
        right_ear = get_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        if avg_ear < EAR_THRESHOLD:
            is_eyes_open = False

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = pose_results.pose_landmarks.landmark

        try:
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x * w, 
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y * h]
            left_eye_p = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x * w, 
                        landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y * h]
            right_eye_p = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x * w, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y * h]
            left_ear_p = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w, 
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h]
            right_ear_p = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * w, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * h]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, 
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, 
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]

            neck_angle = get_vertical_angle(left_ear_p, shoulder)
            torso_angle = get_vertical_angle(shoulder, hip)
            eye_dist = math.dist(left_eye_p, right_eye_p)
            
            dist_nose_left = abs(nose[0] - left_ear_p[0])
            dist_nose_right = abs(nose[0] - right_ear_p[0])
            if dist_nose_right == 0: dist_nose_right = 0.01 
            head_turn_ratio = dist_nose_left / dist_nose_right
            

            if head_turn_ratio > 3.0 or head_turn_ratio < 0.3:
                is_looking = False

            status = "PERFECT POSTURE"
            color = (0, 255, 0)

            if not is_looking:
                status = "NOT LOOKING AT SCREEN (PAUSED)"
                color = (255, 0, 255)
            elif not is_eyes_open:
                status = "EYES CLOSED (PAUSED)"
                color = (255, 0, 255)
            elif eye_dist > 80: 
                status = "WARNING: TOO CLOSE!"
                color = (0, 165, 255)
            elif eye_dist < 40: 
                status = "WARNING: TOO FAR!"
                color = (255, 255, 0)
            elif neck_angle > 30: 
                status = "FORWARD HEAD (TEXT NECK)!"
                color = (0, 0, 255)
            elif torso_angle > 15: 
                status = "SLOUCHING (BACK)!"
                color = (0, 0, 255)

            cv2.putText(frame, status, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (w - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Neck Ang: {int(neck_angle)}", (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Face Ratio: {head_turn_ratio:.1f}", (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if is_looking and is_eyes_open:
                active_screen_time += dt

        except Exception:
            pass
    else:
        cv2.putText(frame, "NO PERSON DETECTED (PAUSED)", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 3)

    mins, secs = divmod(int(active_screen_time), 60)
    cv2.rectangle(frame, (0, h - 40), (250, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Time: {mins:02d}:{secs:02d}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Office Syndrome AI Tracker", frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
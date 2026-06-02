import cv2
import mediapipe as mp
import math
import time
import json
import asyncio
import websockets

# ---------------------------------------------------------
# 1. ฟังก์ชันคำนวณคณิตศาสตร์ (Ergonomics & EAR)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. ตัวแปรเก็บสถานะส่วนกลาง (State) เพื่อส่งให้มาสคอต
# ---------------------------------------------------------
app_state = {
    "status": "STARTING...",
    "is_present": False,
    "is_looking": False,
    "is_eyes_open": False,
    "screen_time": 0,
    "exp": 0,
    "warning_level": 0 # 0=ปกติ, 1=เตือนเบาๆ, 2=เตือนหนัก (เอาไว้เปลี่ยนหน้าตามาสคอต)
}

# ---------------------------------------------------------
# 3. ฟังก์ชัน AI ประมวลผลกล้อง (ทำงานเบื้องหลัง)
# ---------------------------------------------------------
async def process_camera():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(0)
    last_time = time.time()
    
    EAR_THRESHOLD = 0.15

    while True:
        success, frame = cap.read()
        if not success:
            await asyncio.sleep(0.1)
            continue

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pose_results = pose.process(rgb)
        face_results = face_mesh.process(rgb)

        is_looking = True
        is_eyes_open = True
        is_present = False
        current_status = "PERFECT POSTURE"
        warning_level = 0

        # ตรวจจับการหลับตา (Face Mesh)
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            
            left_eye = [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in left_eye_indices]
            right_eye = [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in right_eye_indices]
            
            avg_ear = (get_ear(left_eye) + get_ear(right_eye)) / 2.0
            if avg_ear < EAR_THRESHOLD:
                is_eyes_open = False

        # ตรวจจับท่านั่ง (Pose)
        if pose_results.pose_landmarks:
            is_present = True
            landmarks = pose_results.pose_landmarks.landmark
            try:
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x * w, landmarks[mp_pose.PoseLandmark.NOSE.value].y * h]
                left_eye_p = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y * h]
                right_eye_p = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y * h]
                left_ear_p = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h]
                right_ear_p = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * h]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]

                neck_angle = get_vertical_angle(left_ear_p, shoulder)
                torso_angle = get_vertical_angle(shoulder, hip)
                eye_dist = math.dist(left_eye_p, right_eye_p)
                
                dist_nose_left = abs(nose[0] - left_ear_p[0])
                dist_nose_right = abs(nose[0] - right_ear_p[0]) if abs(nose[0] - right_ear_p[0]) != 0 else 0.01
                head_turn_ratio = dist_nose_left / dist_nose_right
                
                if head_turn_ratio > 3.0 or head_turn_ratio < 0.3:
                    is_looking = False

                # ลอจิกการแจ้งเตือน
                if not is_looking:
                    current_status = "NOT LOOKING AT SCREEN"
                    warning_level = 1
                elif not is_eyes_open:
                    current_status = "EYES CLOSED (PAUSED)"
                    warning_level = 1
                elif eye_dist > 80: 
                    current_status = "WARNING: TOO CLOSE!"
                    warning_level = 2
                elif eye_dist < 40: 
                    current_status = "WARNING: TOO FAR!"
                    warning_level = 1
                elif neck_angle > 30: 
                    current_status = "FORWARD HEAD (TEXT NECK)!"
                    warning_level = 2
                elif torso_angle > 15: 
                    current_status = "SLOUCHING (BACK)!"
                    warning_level = 2
                else:
                    # ถ้านั่งตัวตรง มองจอ ให้ EXP เพิ่มขึ้นเรื่อยๆ
                    app_state["exp"] += (10 * dt) # เพิ่ม 10 EXP ต่อวินาที (จำลอง)

            except Exception:
                pass
        else:
            current_status = "NO PERSON DETECTED"
            warning_level = 0

        # อัปเดตเวลา Screen Time
        if is_looking and is_eyes_open and is_present:
            app_state["screen_time"] += dt

        # อัปเดตสถานะส่วนกลาง
        app_state["status"] = current_status
        app_state["is_present"] = is_present
        app_state["is_looking"] = is_looking
        app_state["is_eyes_open"] = is_eyes_open
        app_state["warning_level"] = warning_level

        # --- สำคัญ: ปล่อยให้ลูป Async เดินหน้า ---
        # (เราเอา cv2.imshow ออก เพื่อให้มันรันอยู่เบื้องหลังแบบไม่มีหน้าต่าง)
        await asyncio.sleep(0.01)

# ---------------------------------------------------------
# 4. ฟังก์ชัน WebSocket (ทำหน้าที่ส่งข้อมูลให้ UI)
# ---------------------------------------------------------
async def websocket_handler(websocket):
    print("UI Frontend Connected!")
    try:
        while True:
            # แปลงข้อมูลใน Dictionary เป็นข้อความ JSON แล้วส่งออกไป
            payload = json.dumps(app_state)
            await websocket.send(payload)
            # ส่งข้อมูลอัปเดตไปที่ UI ทุกๆ 0.1 วินาที (10 Hz)
            await asyncio.sleep(0.1)
    except websockets.exceptions.ConnectionClosed:
        print("UI Frontend Disconnected.")

# ---------------------------------------------------------
# 5. ฟังก์ชัน Main (สั่งรันทั้ง AI และ WebSocket พร้อมกัน)
# ---------------------------------------------------------
async def main():
    print("Starting Edge AI Backend...")
    # เปิดเซิร์ฟเวอร์ WebSocket ที่พอร์ต 8765
    server = await websockets.serve(websocket_handler, "localhost", 8765)
    print("WebSocket Server running on ws://localhost:8765")
    
    # รันกล้อง AI
    await process_camera()

if __name__ == "__main__":
    asyncio.run(main())
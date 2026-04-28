import cv2
import mediapipe as mp
import numpy as np
import csv

# ==========================================
# ส่วนที่ 1: ตั้งค่า MediaPipe (พร้อมให้ไฟล์อื่นดึงไปใช้งาน)
# ==========================================
# เรียกใช้ MediaPipe
mp_holistic = mp.solutions.holistic # โมเดลตรวจจับใบหน้า มือ และร่างกาย
mp_drawing = mp.solutions.drawing_utils # เครื่องมือวาดเส้นและจุด


# ==========================================
# ส่วนที่ 2: ฟังก์ชันหลัก (พร้อมให้ไฟล์อื่นดึงไปใช้งาน)
# ==========================================
# ฟังก์ชันคำนวณมุม
def cal_angle(a, b, c):
    a = np.array(a) # จุดเริ่ม
    b = np.array(b) # จุดหมุน
    c = np.array(c) # จุดสิ้นสุด
    
    radian = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radian*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# ฟังก์ชันสกัดพิกัด (88 จุด)
def extract_keypoints(results):
    # ไหล่ + ศอก
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        l_sh = [landmarks[11].x, landmarks[11].y] # ไหล่ซ้าย
        l_el = [landmarks[13].x, landmarks[13].y] # ศอกซ้าย
        l_wr = [landmarks[15].x, landmarks[15].y] # มือซ้าย
        
        r_sh = [landmarks[12].x, landmarks[12].y] # ไหล่ขวา
        r_el = [landmarks[14].x, landmarks[14].y] # ศอกขวา
        r_wr = [landmarks[16].x, landmarks[16].y] # มือขวา
        
        # มุมข้อศอกซ้าย, ขวา
        angle_l_elbow = cal_angle(l_sh, l_el, l_wr)
        angle_r_elbow = cal_angle(r_sh, r_el, r_wr)
        
        # มุมหัวไหล่ซ้าย, ขวา 
        l_hip_ref = [l_sh[0], l_sh[1] + 1.0]
        r_hip_ref = [r_sh[0], r_sh[1] + 1.0]
        
        angle_l_shoulder = cal_angle(l_hip_ref, l_sh, l_el)
        angle_r_shoulder = cal_angle(r_hip_ref, r_sh, r_el)
        
        # ยัดใส่ array
        pose_angles = np.array([angle_l_elbow, angle_r_elbow, angle_l_shoulder, angle_r_shoulder])
    else:
        pose_angles = np.zeros(4) # จับไม่เจอใส่ 0

    # มือซ้าย 
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 2) # จับไม่เจอใส่ 0

    # มือขวา
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 2) # จับไม่เจอใส่ 0

    # เอามาต่อกัน (4 + 42 + 42 = 88 var/frame)
    return np.concatenate([pose_angles, lh, rh])


# ==========================================
# ส่วนที่ 3: ลูปทดสอบและการเก็บข้อมูล (รันเฉพาะเมื่อกดรันไฟล์นี้โดยตรง)
# ==========================================
def run_data_collection():
    csv_file = 'sign_language_data.csv'
    is_recording = False

    # เปิดกล้อง
    cap = cv2.VideoCapture(0) 

    # ตั้งค่าการทำงานของโมเดล
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, # ความมั่นใจขั้นต่ำในการตรวจจับคน
        min_tracking_confidence=0.5   # ความมั่นใจขั้นต่ำในการแทร็กการเคลื่อนไหว
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # ประมวลผลภาพ
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # แสดง keypoints ในกล้อง
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            if is_recording:
                # ดึงจุดมาเป็นตัวเลข
                keypoints = extract_keypoints(results)
                
                # เขียนลงไฟล์ CSV
                with open(csv_file, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(keypoints.tolist())
                
                cv2.putText(image, 'RECORDING...', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, 'Press "R" to Record / "S" to Stop', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('SignBridge - Data Collection', image)

            # รอรับคำสั่งจากคีย์บอร์ด
            key = cv2.waitKey(10) & 0xFF
            if key == ord('r'):
                is_recording = True
                print("เริ่มบันทึกข้อมูล...")
            elif key == ord('s'):
                is_recording = False
                print("หยุดบันทึกข้อมูล...")
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# โค้ดส่วนนี้จะทำงานก็ต่อเมื่อคุณสั่งรันไฟล์นี้โดยตรง (ไม่ได้โดน import)
if __name__ == "__main__":
    run_data_collection()
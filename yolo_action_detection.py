import cv2
from ultralytics import YOLO
import numpy as np
import pyttsx3
import time
import json



# FPS & LOGGING INITIALIZATION


fps_prev_time = time.time()
fps_display = 0

# Logging lists for graphs
fps_log = []
time_log = []

action_log = []
frame_log = []

knee_angle_log = []
movement_log = []



# TEXT-TO-SPEECH INITIALIZATION

try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 170)
except Exception as e:
    print(f"Error initializing pyttsx3: {e}")
    tts_engine = None


#STATE MACHINE VARIABLES

last_spoken_action = ""
current_locked_action = "DETECTING..."
last_detected_action = "DETECTING..."
consecutive_frame_count = 0
previous_hip_center = None

REQUIRED_FRAMES = 8                # → prevents flickering
MOTION_THRESHOLD_PIXELS = 10       # → walking detection threshold


# LOAD MODEL


model = YOLO("yolov8s-pose.pt")
cap = cv2.VideoCapture(0)

# HELPERS

def get_angle(a, b, c):
    """Calculate the angle between three keypoints."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def classify_static_action(keypoints):
    """Classify Standing / Sitting / Kneeling using knee angles."""

    R_HIP = keypoints[12]
    R_KNEE = keypoints[14]
    R_ANK = keypoints[16]

    left_knee_angle = get_angle(keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])
    right_knee_angle = get_angle(R_HIP[:2], R_KNEE[:2], R_ANK[:2])
    avg_knee = (left_knee_angle + right_knee_angle) / 2

    knee_angle_log.append(avg_knee)

    knee_ankle_dist = np.linalg.norm(np.array(R_KNEE[:2]) - np.array(R_ANK[:2]))
    hip_knee_dist = np.linalg.norm(np.array(R_HIP[:2]) - np.array(R_KNEE[:2]))

    if avg_knee > 165:
        return "Standing"
    if 70 < avg_knee < 120:
        return "Sitting"
    if 10 < avg_knee < 70 and knee_ankle_dist < (0.8 * hip_knee_dist):
        return "Kneeling"

    return "Unclassified"


def announce_action(action):
    """Speaks the action aloud if changed."""
    global last_spoken_action

    if tts_engine and action != last_spoken_action:
        spoken = action.replace("Unclassified", "Unclassified Pose")
        tts_engine.say(spoken)
        tts_engine.runAndWait()
        last_spoken_action = action



# MAIN LOOP


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS CALCULATION 
    now = time.time()
    fps_display = round(1 / (now - fps_prev_time + 1e-6), 1)
    fps_prev_time = now

    # Log FPS
    fps_log.append(fps_display)
    time_log.append(now)

    # YOLO POSE
    results = model(frame, verbose=False)
    action_for_display = current_locked_action

    if results and results[0].keypoints is not None and results[0].keypoints.xy.numel() > 0:

        keypoints = results[0].keypoints.xy.tolist()[0]

        if np.sum(keypoints[11]) > 0 and np.sum(keypoints[12]) > 0:

            # MOTION TRACKING 
            L_HIP = keypoints[11]
            R_HIP = keypoints[12]

            center = np.array([(L_HIP[0] + R_HIP[0]) / 2, (L_HIP[1] + R_HIP[1]) / 2])

            movement = 0
            if previous_hip_center is not None:
                movement = np.linalg.norm(center - previous_hip_center)

            movement_log.append(movement)
            previous_hip_center = center

            # POSE CLASSIFICATION 
            current_action = classify_static_action(keypoints)

            # Movement only counts as walking if standing
            if movement > MOTION_THRESHOLD_PIXELS and current_action == "Standing":
                current_action = "Walking"

            # STATE MACHINE
            if current_action == current_locked_action:
                consecutive_frame_count = 0
            elif current_action == last_detected_action:
                consecutive_frame_count += 1
                if consecutive_frame_count >= REQUIRED_FRAMES:
                    current_locked_action = current_action
                    consecutive_frame_count = 0
            else:
                consecutive_frame_count = 1
                last_detected_action = current_action

            # LOG ACTION
            action_log.append(current_action)
            frame_log.append(len(frame_log))

            # DISPLAY TEXT LOGIC
            if current_locked_action in ["Unclassified", "DETECTING..."]:
                action_for_display = "Processing..."
            else:
                action_for_display = current_locked_action
                announce_action(current_locked_action)

            # Draw keypoints
            for x, y in [kp[:2] for kp in keypoints]:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)


    # DISPLAY: ACTION + FPS


    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)

    # Action text
    action_text = f"Action: {action_for_display}"
    cv2.putText(frame, action_text, (30, 50), font, 1.5, color, 3)

    # FPS text
    fps_text = f"FPS: {fps_display}"
    cv2.putText(frame, fps_text, (30, 110), font, 1.2, (0, 255, 255), 3)

    cv2.imshow("Pose Estimation with Action Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



# SAVE LOG FILE FOR GRAPHING

log_data = {
    "fps": fps_log,
    "time": time_log,
    "actions": action_log,
    "frames": frame_log,
    "knee_angles": knee_angle_log,
    "movement": movement_log
}

with open("pose_logs.json", "w") as f:
    json.dump(log_data, f, indent=4)

print("\n Log file saved as pose_logs.json\n")

# Cleanup
if tts_engine:
    tts_engine.stop()

cap.release()
cv2.destroyAllWindows()

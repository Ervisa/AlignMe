import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import time
from collections import deque

# Pilates Form Feedback with Smoothing and Two-Point Calibration (Arm)
# Tracks RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Text-to-speech engine setup.
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Calibration and feedback thresholds.
angle_A = None   # Endpoint A (e.g., full extension)
angle_B = None   # Endpoint B (e.g., full bend)
form_threshold = 70       # % below which prompts 'Adjust your form'
bend_threshold = 20       # % for 'Good bend'
straight_threshold = 80   # % for 'Good extension'

# Smoothing buffer for angle values.
buffer_size = 5
angle_buffer = deque(maxlen=buffer_size)

# Track last feedback to avoid repetition.
last_feedback = None

print("Press 'e' to set endpoint A (full extension).")
print("Press 'b' to set endpoint B (full bend).")
print("Press 'q' to quit.")

# Helper: compute angle at point b given points a-b-c.
def get_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# Start video capture.
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Keyboard input once per frame.
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    raw_angle = None
    score = None
    feedback_text = None

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark

        # Arm landmarks: right shoulder, elbow, wrist.
        shoulder = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
        elbow    = (lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h)
        wrist    = (lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h)

        # Compute raw elbow angle.
        raw_angle = get_angle(shoulder, elbow, wrist)
        # Append to smoothing buffer.
        angle_buffer.append(raw_angle)
        # Compute smoothed angle.
        angle = sum(angle_buffer) / len(angle_buffer)

        # Calibration: set endpoints A/B.
        if key == ord('e') and angle is not None:
            angle_A = angle
            print(f"Endpoint A set: {angle_A:.1f}°")
            engine.say("Endpoint A set")
            engine.runAndWait()
        elif key == ord('b') and angle is not None:
            angle_B = angle
            print(f"Endpoint B set: {angle_B:.1f}°")
            engine.say("Endpoint B set")
            engine.runAndWait()

        # Compute form score if both endpoints defined and distinct.
        if angle_A is not None and angle_B is not None and abs(angle_A - angle_B) > 1e-3:
            # Determine min/max so mapping works irrespective of calibration order.
            angle_min = min(angle_A, angle_B)
            angle_max = max(angle_A, angle_B)
            # Linear score mapping [angle_min→0%, angle_max→100%]
            score = (angle - angle_min) / (angle_max - angle_min) * 100
            score = max(0.0, min(100.0, score))

            # Determine feedback text based on score bands.
            if score <= bend_threshold:
                feedback_text = "Good bend"
            elif score >= straight_threshold:
                feedback_text = "Good extension"
            elif score < form_threshold:
                feedback_text = "Adjust your form"

            # Speak feedback only if it changed.
            if feedback_text and feedback_text != last_feedback:
                engine.say(feedback_text)
                engine.runAndWait()
                last_feedback = feedback_text

            # Draw color-coded arm segments based on smoothed angle.
            color = (0, 255, 0) if score >= form_threshold else (0, 0, 255)
            cv2.line(frame, (int(shoulder[0]), int(shoulder[1])),
                     (int(elbow[0]), int(elbow[1])), color, 4)
            cv2.line(frame, (int(elbow[0]), int(elbow[1])),
                     (int(wrist[0]), int(wrist[1])), color, 4)
            cv2.circle(frame, (int(elbow[0]), int(elbow[1])), 8, color, cv2.FILLED)

        # Overlay text: raw angle, smoothed angle, endpoints, and form score.
        if raw_angle is not None:
            cv2.putText(frame, f"Raw: {int(raw_angle)}°", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if angle is not None:
            cv2.putText(frame, f"Smooth: {int(angle)}°", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        if angle_A is not None:
            cv2.putText(frame, f"A: {int(angle_A)}°", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        if angle_B is not None:
            cv2.putText(frame, f"B: {int(angle_B)}°", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        if score is not None:
            cv2.putText(frame, f"Form: {int(score)}%", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if score >= form_threshold else (0, 0, 255), 2)
        if feedback_text:
            cv2.putText(frame, feedback_text, (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Form Feedback (Arm)", frame)

# Cleanup
cap.release()
cv2.destroyAllWindows()


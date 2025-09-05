import pathlib
import cv2
import mediapipe as mp
from helpers import all_data_to_csv as dtc
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

current_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())

# ===========================================================
# ======================= SETTINGS ==========================
show_video = True
flip_image = False  # Set to True if you need to flip the video

cap = cv2.VideoCapture("../demo_data/video_rotate.mp4")  # Video

result_csv_filename = f"../demo_data/csv_results/csv_file_{current_time}.csv"
# ===========================================================

csv_writer = dtc.CSVDataWriter()
success = True

with (mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose,
      mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands):
    while cap.isOpened() and success:
        success, image = cap.read()
        if not success:
            break

        if flip_image:
            image = cv2.flip(image, 1)

        # Convert to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        pose_results = pose.process(image)
        hands_results = hands.process(image)

        # Draw annotations if showing video
        if show_video:
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw pose landmarks
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Draw hand landmarks
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                    )

            cv2.imshow('MediaPipe Pose + Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

        # Process data - pass both pose and hands results
        csv_writer.read_data(
            pose_data=pose_results.pose_landmarks,
            hands_data=hands_results.multi_hand_landmarks,
            hand_labels=hands_results.multi_handedness,
            timestamp=cap.get(cv2.CAP_PROP_POS_MSEC)
        )

csv_writer.to_csv(result_csv_filename)
cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import yaml
import pathlib

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

script_dir = pathlib.Path(__file__).parent

show_video = True
show_data = True
flip_image = False  # when your webcam flips your image, you may need to re-flip it by setting this to True

cap = cv2.VideoCapture(index=0)  # Live from camera

with open(script_dir.joinpath("keypoint_mapping.yml"), "r") as yaml_file:
    mappings = yaml.safe_load(yaml_file)
    KEYPOINT_NAMES = mappings["face"] + mappings["body"] + mappings["left_hand"] + mappings["right_hand"]

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

        # Back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape  # for coordinate scaling

        # --- Show Pose Data ---
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(image, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, "Pose detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # --- Show Hand Data ---
        if hands_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.putText(image, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, "Hand detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2, cv2.LINE_AA)

        if show_video:
            cv2.imshow('MediaPipe Pose + Hands', image)

        # press ESC to stop the loop
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

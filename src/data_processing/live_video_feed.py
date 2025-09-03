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

# cap = cv2.VideoCapture(filename=str(script_dir.joinpath("../demo_data/video_rotate.mp4"))) # Video
cap = cv2.VideoCapture(index=0)  # Live from camera (change index if you have more than one camera)

with open(script_dir.joinpath("keypoint_mapping.yml"), "r") as yaml_file:
    mappings = yaml.safe_load(yaml_file)
    KEYPOINT_NAMES = mappings["face"]
    KEYPOINT_NAMES += mappings["body"]
    KEYPOINT_NAMES += mappings["left_hand"]
    KEYPOINT_NAMES += mappings["right_hand"]

success = True

with (mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose,
      mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands):

    while cap.isOpened() and success:
        success, image = cap.read()
        if not success:
            break

        if flip_image:
            image = cv2.flip(image, 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        pose_results = pose.process(image)
        hands_results = hands.process(image)

        #Show Pose Data
        if pose_results.pose_landmarks:

        #Show hand Data
        # press ESC to stop the loop
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()

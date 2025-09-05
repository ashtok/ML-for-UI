import pandas as pd
import yaml
import pathlib


class CSVDataWriter:
    def __init__(self):
        self.frame_list = []
        self.timestamps = []
        self.column_names = self.load_keypoint_mapping_from_file("./keypoint_mapping.yml")

    def read_data(self, pose_data=None, hands_data=None, hand_labels=None, timestamp=None):
        """
        Process pose and hands data and add to frame list

        Args:
            pose_data: MediaPipe pose landmarks
            hands_data: MediaPipe multi_hand_landmarks
            hand_labels: MediaPipe multi_handedness (to identify left/right hands)
            timestamp: Video timestamp
        """
        frame = []

        # Process pose data (33 landmarks, 4 values each: x, y, z, visibility)
        if pose_data is not None:
            for i in range(33):
                frame.append(pose_data.landmark[i].x)
                frame.append(pose_data.landmark[i].y)
                frame.append(pose_data.landmark[i].z)
                frame.append(pose_data.landmark[i].visibility)
        else:
            # Fill with NaN if no pose detected
            frame.extend([float('nan')] * (33 * 4))

        # Initialize hand data arrays (21 landmarks, 3 values each: x, y, z)
        left_hand_data = [float('nan')] * (21 * 3)
        right_hand_data = [float('nan')] * (21 * 3)

        # Process hands data
        if hands_data is not None and hand_labels is not None:
            for hand_landmarks, handedness in zip(hands_data, hand_labels):
                hand_label = handedness.classification[0].label  # 'Left' or 'Right'

                # Extract landmark data
                hand_coords = []
                for landmark in hand_landmarks.landmark:
                    hand_coords.extend([landmark.x, landmark.y, landmark.z])

                # Assign to correct hand
                if hand_label == 'Left':
                    left_hand_data = hand_coords
                elif hand_label == 'Right':
                    right_hand_data = hand_coords

        # Add hand data to frame (left hand first, then right hand)
        frame.extend(left_hand_data)
        frame.extend(right_hand_data)

        self.frame_list.append(frame)
        self.timestamps.append(timestamp)
        return frame

    def to_csv(self, output_path):
        """Save data to CSV file"""
        frames = pd.DataFrame(self.frame_list, columns=self.column_names, index=self.timestamps)
        frames.index.name = "timestamp"
        frames.index = frames.index.astype(int)
        frames.round(5).to_csv(output_path)
        print(f"Data saved to: {output_path}")
        print(f"Shape: {frames.shape}")

    def load_keypoint_mapping_from_file(self, file):
        """Load keypoint names from YAML file and create column names"""
        script_dir = pathlib.Path(__file__).parent.parent
        yaml_path = script_dir.joinpath(file)

        with open(yaml_path, "r") as yaml_file:
            mappings = yaml.safe_load(yaml_file)

            # Get all keypoint names in the correct order
            face_names = mappings.get("face", [])
            body_names = mappings.get("body", [])
            left_hand_names = mappings.get("left_hand", [])
            right_hand_names = mappings.get("right_hand", [])

            column_names = []

            # Add pose columns (face + body landmarks with x, y, z, visibility)
            pose_names = face_names + body_names
            for joint_name in pose_names:
                column_names.extend([
                    f"{joint_name}_x",
                    f"{joint_name}_y",
                    f"{joint_name}_z",
                    f"{joint_name}_visibility"
                ])

            # Add left hand columns (x, y, z only)
            for joint_name in left_hand_names:
                column_names.extend([
                    f"{joint_name}_x",
                    f"{joint_name}_y",
                    f"{joint_name}_z"
                ])

            # Add right hand columns (x, y, z only)
            for joint_name in right_hand_names:
                column_names.extend([
                    f"{joint_name}_x",
                    f"{joint_name}_y",
                    f"{joint_name}_z"
                ])

            return column_names
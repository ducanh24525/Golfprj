import pickle
import cv2
import os

pose_file = 'golfdb_pose.pkl'
with open(pose_file, 'rb') as f:
    pose_dict = pickle.load(f)

# Get the first video id
video_id = 30
poses = pose_dict[video_id]

print(f"Video {video_id} poses shape: {poses.shape}")

# Create an image to draw keypoints
vid_path = f"videos_160/videos_160/{video_id}.mp4"
if os.path.exists(vid_path):
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    if ret:
        frame_poses = poses[0]
        # Draw the 13 points with their indices
        for idx, pt in enumerate(frame_poses):
            x, y, conf = pt
            if conf > 0:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                cv2.putText(frame, str(idx), (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imwrite("first_frame_keypoints.jpg", frame)
        print("Saved first_frame_keypoints.jpg")
    cap.release()

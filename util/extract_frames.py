import cv2
import os

video_dir = '../../data/raw/parking/normal/'

video_extensions = ('.mp4',)

for file_name in os.listdir(video_dir):
    if file_name.lower().endswith(video_extensions):
        video_path = os.path.join(video_dir, file_name)
        
        video_name = os.path.splitext(file_name)[0]
        
        cap = cv2.VideoCapture(video_path)
        count = 0
        
        print(f"▶ [{file_name}] extract start ...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 30프레임당 1장씩 저장
            if count % 30 == 0:
                frame_name = f"{video_name}_frame_{count}.jpg"
                frame_path = os.path.join(video_dir, frame_name)
                cv2.imwrite(frame_path, frame)
            
            count += 1
            
        cap.release()
        print(f" >> single finished")

print("\n all finished!")
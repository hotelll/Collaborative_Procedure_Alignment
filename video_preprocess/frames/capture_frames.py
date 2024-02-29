import cv2
import os
from tqdm import tqdm


video_root = '/mnt/extended/hotel/CSV/videos'
frame_root = '/mnt/extended/hotel/CSV/frames'

class_list = os.listdir(video_root)

for cls_id in tqdm(class_list):
    cls_dir = os.path.join(video_root, cls_id)
    video_list = os.listdir(cls_dir)
    for video_id in video_list:
        video_path = os.path.join(cls_dir, video_id)

        frame_dir = os.path.join(os.path.join(frame_root, cls_id), video_id[:-4])

        os.makedirs(frame_dir, exist_ok=True)

        video_capture = cv2.VideoCapture(video_path)

        number = 1
        false_num = 0
        while True:
            flag, frame = video_capture.read()
            
            frame_path = os.path.join(frame_dir, str(number)+'.jpg')
            if flag:
                frame = cv2.resize(frame, (320, 180), cv2.INTER_AREA)
                cv2.imwrite(frame_path, frame)
                number += 1
            else:
                false_num += 1
                if false_num == 20:
                    break
                continue

        video_capture.release()

# '/mnt/extended/hotel/CSV/videos/1.2/chenyijun.MP4'
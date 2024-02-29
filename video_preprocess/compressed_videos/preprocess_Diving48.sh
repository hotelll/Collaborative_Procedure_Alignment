#!/bin/bash
set -x
set -p

for i in $(ls /mnt/extended/hotel/Diving48/videos); do

OUTPUT_DIR=$(echo $i | sed s/\.mp4// | sed s/\.MP4//)
# echo $OUTPUT_DIR
mkdir -p /mnt/extended/hotel/Diving48/video_compressed/
ffmpeg -i /mnt/extended/hotel/Diving48/videos/$i -s 320x180 -pix_fmt yuv420p -crf 18 -tune fastdecode -y /mnt/extended/hotel/Diving48/video_compressed/$OUTPUT_DIR.mp4

done

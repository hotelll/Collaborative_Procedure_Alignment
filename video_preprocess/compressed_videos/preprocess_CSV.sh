#!/bin/bash
set -x
set -p

for i in $(ls /mnt/extended/hotel/CSV/videos); do
for j in $(ls /mnt/extended/hotel/CSV/videos/$i); do

OUTPUT_DIR=$(echo $j | sed s/\.mp4// | sed s/\.MP4//)
# echo $OUTPUT_DIR
mkdir -p /mnt/extended/hotel/CSV/video_compressed/$i
ffmpeg -i /mnt/extended/hotel/CSV/videos/$i/$j -s 320x180 -pix_fmt yuv420p -crf 18 -tune fastdecode -y /mnt/extended/hotel/CSV/video_compressed/$i/$OUTPUT_DIR.mp4

done
done

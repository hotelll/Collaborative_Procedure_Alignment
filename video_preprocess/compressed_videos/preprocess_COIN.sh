#!/bin/bash
set -x
set -p

for i in $(ls /mnt/extended/hotel/COIN-SV/videos); do
for j in $(ls /mnt/extended/hotel/COIN-SV/videos/$i); do

OUTPUT_DIR=$(echo $j | sed s/\.mp4// | sed s/\.MP4//)
# echo $OUTPUT_DIR
mkdir -p /mnt/extended/hotel/COIN-SV/video_compressed/
ffmpeg -i /mnt/extended/hotel/COIN-SV/videos/$i/$j -s 320x180 -pix_fmt yuv420p -crf 18 -tune fastdecode -y /mnt/extended/hotel/COIN-SV/video_compressed/$OUTPUT_DIR.mp4

done
done

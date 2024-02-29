#!/bin/bash
set -x
set -p

for i in $(ls /mnt/extended/hotel/COIN-SV/video_compressed); do

OUTPUT_DIR=$(echo $i | sed s/\.mp4// | sed s/\.MP4//)
# echo $OUTPUT_DIR
mkdir -p /mnt/extended/hotel/COIN-SV/frames/$OUTPUT_DIR
ffmpeg -i /mnt/extended/hotel/COIN-SV/video_compressed/$i -s 320x180 -q:v 1 -qmin 1 -qmax 1 /mnt/extended/hotel/COIN-SV/frames/$OUTPUT_DIR/%d.jpg

done

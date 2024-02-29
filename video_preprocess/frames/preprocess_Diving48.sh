#!/bin/bash
set -x
set -p

for i in $(ls /mnt/extended/hotel/Diving48/videos); do

OUTPUT_DIR=$(echo $i | sed s/\.mp4// | sed s/\.MP4//)
# echo $OUTPUT_DIR
mkdir -p /mnt/extended/hotel/Diving48/frames/$OUTPUT_DIR
ffmpeg -i /mnt/extended/hotel/Diving48/videos/$i -s 320x180 -y /mnt/extended/hotel/Diving48/frames/$OUTPUT_DIR/%06d.jpg

done

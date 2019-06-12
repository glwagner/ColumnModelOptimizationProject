#!/bin/bash

framename="$1_%04d.png"
filename="$1.mp4"

# Options
framerate=8
nframes=58

ffmpeg \
    -r $framerate \
    -f image2 \
    -s 1920x1080 \
    -i $framename \
    -vcodec libx264 \
    -crf 25 \
    -pix_fmt yuv420p \
    $filename
#    -vframes $nframes \

# Additional ffmpeg Options
# limit number of frames:   -vframes $nframes \

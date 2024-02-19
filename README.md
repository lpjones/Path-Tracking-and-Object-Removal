Completed by lpjones and ryuliou
# Path Tracking & Object Removal
# Goals
Track people's positions as they move. Then output a video displaying the path people have traveled.
![](out_imgs/Path-find.gif)

# How to use
Download yolov3.weights file at https://pjreddie.com/media/files/yolov3.weights and move the file to the yolo_files/ folder

The two executable files are yolo_path_tracker.py and yolo_remover.py

yolo_path_tracker.py:

The input file is input.webm and the output file is output.mp4.

yolo_path_tracker.py takes the video and tracks people in the video across different frames maintaining the same ID per person along with a trail behind them to see how they have traveled based on their centroid.

Command to execute:
```console
python3 yolo_path_tracker.py
```
Libraries used: \
opencv2 \
numpy \
pandas \
time

yolo_remover.py:

Takes in class to remove, size of chunk to replace class with, and images to modify

Each image finds the bounding box for the class given and copies the given chunk size to the right of the bounding box and copies it over the bounding box removing the original pixels within the bounding box.

Command to execute yolo_remover.py using class of person, chunk size of 10 and modifying all images in imgs/:
```console
python3 yolo_remover.py person 10 imgs/
```
Libraries used: \
os \
opencv2 \
numpy \
matplotlib \
argparse \
pandas \
time

import cv2
import numpy as np
from tracker import Tracker
# Added:
import pandas as pd
import time


# Load YOLO model
net = cv2.dnn.readNetFromDarknet("yolo_files/yolov3.cfg","yolo_files/yolov3.weights")
yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']

# Load video
video_path = 'input.webm'
cap = cv2.VideoCapture(video_path)

# Initialize the tracker
tracker = Tracker()

# Determine the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set up the video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
out_vid = cv2.VideoWriter('output.mp4', fourcc, 15.0, (int(cap.get(3)), int(cap.get(4))))

# Read the coconames label file in
labels = pd.read_csv("yolo_files/coco.names", header=None)

def im2Blob(im): # Referenced From Lab A1
    input_blob = cv2.dnn.blobFromImage(im, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    return [input_blob]

def imgs_to_network(blob_imgs, yolo_layers, network): # Referenced from lab A1
    outputs = []
    times = []
    for im in blob_imgs:
        #print(np.array(im).shape)
        network.setInput(im)
        start_time = time.time()
        output = network.forward(yolo_layers)
        end_time = time.time()
        times.append(end_time - start_time)
        outputs.append(output)
    return outputs, sum(times) / len(times)

def get_bounding_boxes(image, output): # Referenced from lab A1
    # Define variables for drawing on image
    bounding_boxes = []
    confidences = []
    classes = []
    probability_minimum = 0.5
    threshold = 0.3
    h, w = image.shape[:2]
    # Get bounding boxes, confidences and classes
    for result in output:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                classes.append(class_current)

    return bounding_boxes, classes, confidences, probability_minimum, threshold

def apply_nms(labels, classes, confidences, probability_minimum, threshold):

    # Draw bounding boxes and information on images
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold) # NMSBoxes filters out duplicate boxes

    coco_labels = 80
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')

    filtered_bound_boxes = [] # Only contains bounding boxes with people in them
    if len(results) > 0:
        for i in results.flatten(): 
            if labels[0][classes[i]] == 'person': 
                filtered_bound_boxes.append(bounding_boxes[i])

    # Convert format to put into trackers update function
    for filtered_box in filtered_bound_boxes:
        filtered_box[2] = filtered_box[0] + filtered_box[2]
        filtered_box[3] = filtered_box[1] + filtered_box[3]
    
    return filtered_bound_boxes

def draw_bound_boxes(image, trackers, max):
    for i in range(len(trackers)):

        x_min, y_min = int(trackers[i].bounding_box[2]), int(trackers[i].bounding_box[3])
        x_max, y_max = int(trackers[i].bounding_box[0]), int(trackers[i].bounding_box[1])

        colour_box = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 0, 0), (0, 255, 0)]
        col_box = colour_box[trackers[i].id % 5]
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), col_box, 2)
        cv2.circle(image, trackers[i].centroid, 5, col_box, 5)

        # TODO: Draw Centroid and put text above that
        text_box = '{}'.format(int(trackers[i].id)) # Print out ID of tracked person

        cv2.putText(image, text_box, (trackers[i].centroid[0] - 10, trackers[i].centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .75, col_box, 2)
        
        paths = trackers[i].paths[-10:]
        for i in range(len(paths)-1):
            cv2.line(image, paths[i], paths[i+1], col_box, 3)

    max = tracker.cur_id
        
    #person_counter_text_box =  'People: {}'.format(max)
    # cv2.putText(image, person_counter_text_box, (np.array(video_frames)[0].shape[1]//2-20, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,255,0), 2)
    return image, max 

def centroid_calc(filtered_bound_boxes): # Put centroids found in the current frame into an array
    centroids_in_img =  []
    for box in filtered_bound_boxes:
        x = box[0] + (box[2]-box[0])//2
        y = box[1] + (box[3]-box[1])//2
        centroids_in_img.append((x, y))
    return centroids_in_img

# Process each frame
for _ in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    # - Convert frame to blob
    frame_blob = im2Blob(frame)
    # - Use YOLO net to detect objects
    outputs, avg_time = imgs_to_network(frame_blob, yolo_layers, net) # Inferences are the outputs
    bounding_boxes, classes, confidences, probability_minimum, threshold = get_bounding_boxes(frame, outputs[0])
    # - Apply Non-Maximum Suppression (NMS) to filter out overlapping boxes
    filtered_bound_boxes = apply_nms(labels, classes, confidences, probability_minimum, threshold)

    # - Update the tracker with the centroids of detected objects
    centroids_in_img = centroid_calc(filtered_bound_boxes) # Calculate Centroids based on bound boxes

    trackers = tracker.update(centroids_in_img, filtered_bound_boxes)
    # - Draw bounding boxes, trails, and labels on the frame

    boxed_frame, max = draw_bound_boxes(frame, trackers, max)
    # Write the processed frame to the output video
    
    out_vid.write(boxed_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video writer and video capture objects
out_vid.release()
cap.release()
cv2.destroyAllWindows()

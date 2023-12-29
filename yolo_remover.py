import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import time

# Parse in arguments from command line
# Referenced from argparse library
parser = argparse.ArgumentParser(description="Command-Line Parser")

# Define a positional argument, our input_file
parser.add_argument('class_2_remove', nargs=1, type=str, help='class_name (person)')
parser.add_argument('block_size', nargs=1, type=int, help='Size of Block Blur')
parser.add_argument('input_files', nargs='+', type=str, help='<Images/Image dir>')

args = parser.parse_args()

def check_input_paths(args):
    imgs = []

    # Check for correct num of arguments
    if len(args.input_files) < 1:
        print("Invalid number of input files. Need at least 1\n \
                <Image/Images path>")
        exit(0)

    # Check if input image path is a directory
    if os.path.isdir(args.input_files[0]):
        # If it is a directory pull out images from directory
        for file in os.listdir(args.input_files[0]):
            if file.endswith('.jpg'):
                imgs.append(args.input_files[0] + file)
    else:
        # Check for valid file paths
        for file in args.input_files:
            if not os.path.isfile(file):
                print(f"{file} is not a valid file path")
                exit(0)
        imgs = args.input_files[0:]

    labels = pd.read_csv('yolo_files/coco.names', header=None)
    if args.class_2_remove[0] not in np.array(labels):
        print("Invalid class to Remove")
        exit(0)
    
    return imgs, labels

# Load YOLO model
def load_YOLO(model_architecture, model_weights):
    network = cv2.dnn.readNetFromDarknet(model_architecture, model_weights)
    yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']
    return yolo_layers, network


# Check if image file is valid
def read_img(image_path_arr):
    fig, ax = plt.subplots(1, len(image_path_arr), figsize=(20, 20), squeeze=False)
    images_cv2 = []
    for i, image_path in enumerate(image_path_arr):
        try:
            image = cv2.imread(image_path)
            images_cv2.append(image)
            ax[0][i].axis('off')
            ax[0][i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        except:
            print(f"Invalid Input Image {image_path}")
            exit(0)
    
    #plt.show()
    return images_cv2


# Convert Images to Blobs
def im2Blob(images_cv2):
    imgs_blob = []
    for im in images_cv2:
        input_blob = cv2.dnn.blobFromImage(im, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        imgs_blob.append(input_blob)
    return imgs_blob


# Pass images through the network
def imgs_to_network(blob_imgs, yolo_layers, network):
    outputs = []
    times = []
    for im in blob_imgs:
        network.setInput(im)
        start_time = time.time()
        output = network.forward(yolo_layers)
        end_time = time.time()
        times.append(end_time - start_time)
        outputs.append(output)
    return outputs, sum(times) / len(times)



def draw_image(image, output, labels):
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

    # Draw bounding boxes and information on images
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
    coco_labels = 80
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')
    # Track number of each class predicted per image
    boxes = []
    if len(results) > 0:
        for i in results.flatten(): # i is the value, , not the index of the results array
            if labels[0][classes[i]] == args.class_2_remove[0]:
                boxes.append(bounding_boxes[i])     
    return image, boxes 

def inpaint_yolo_box(image, yolo_boxes, block):
    """
    Inpaints specified bounding boxes in the image using a right-to-left copy method.

    Parameters:
    image (numpy.ndarray): The original image.
    yolo_boxes (list of tuples): List of bounding boxes, each defined as (x, y, width, height).
    block: BLOCK_SIZExBLOCK_SIZE box to copy over.

    Returns:
    The inpainted image.
    """
    for box in yolo_boxes:
        x_min, y_min, width, height = box[0], box[1], box[2], box[3]
        x_right = x_min + width # Rightmost column inside bounding box
        _block = block
        if _block > width:
            _block = width
        for r in range(y_min, y_min + height): # For # of rows
            chunk = []
            try:
                chunk = image[r, x_right+1:x_right+_block+1] # Get chunk to right of bounding box
                #print("Shape:", np.array(chunk).shape)
            except:
                pass
            if len(chunk) != _block:
                _block = len(chunk)
            for c in range(x_right+1, x_min + _block, -_block): # From the right to the x_min
                image[r, (c - _block):c] = chunk # Replace image with chunk
            # For the last iteration, modulo so that it fits in bounding box properly:
            # (width % block) is leftover columns,  -(leftover_cols) is how much from the right of the chunk you should copy

            leftover = width % _block
            if leftover != 0:
                image[r, x_min:(x_min + leftover + 1)] = chunk[-leftover - 1:]
    
    return image

imgs, labels = check_input_paths(args)

yolo_layers, network = load_YOLO('yolo_files/yolov3.cfg', 'yolo_files/yolov3.weights')
imgs_cv2 = read_img(imgs)
imgs_blob = im2Blob(imgs_cv2)

outputs, avg_time = imgs_to_network(imgs_blob, yolo_layers, network) # Inferences are the outputs

output_path = './out_imgs/'
# Create out_images/ if it doesn't exist
try:
    os.mkdir(output_path)
except:
    pass
dicts, img_names = [], []
classes_all_dict = {} # Shows total class instances across all images
# save images with boxes as files in out_images/
for i, im in enumerate(imgs_cv2):
    im, boxes = draw_image(im, outputs[i], labels) # Draw inferences made in imgs_to_network() to the original images (imgs_cv2)
    img = inpaint_yolo_box(im, boxes, args.block_size[0])
    # Extract file name with extension
    file = str(os.path.basename(imgs[i]))
    img_names.append(file)
    # Remove extension
    file = str(os.path.splitext(file)[0])
    # Save to file
    plt.imsave(output_path + file + '_out.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

##from IPython import get_ipython
##
##
##get_ipython().system('pip install -U --pre tensorflow=="2.*"')
##get_ipython().system('pip install tf_slim')
import os
import pathlib

import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from grabscreen import grab_screen
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display


# Import the object detection module.

from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util


# Patches:

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing the path.
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# ## Loader


def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))

  return model


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# For the sake of simplicity we will test on 2 images:

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
TEST_IMAGE_PATHS


# # Detection

# Load an object detection model:

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)


# Check the model's input signature, it expects a batch of 3-color images of type uint8:

print(detection_model.signatures['serving_default'].inputs)


# And returns several outputs:

detection_model.signatures['serving_default'].output_dtypes
detection_model.signatures['serving_default'].output_shapes


# Add a wrapper function to call the model, and cleanup the outputs:

def run_inference_for_single_image(model, image):
  image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = np.expand_dims(image_np, axis=0)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


# Run it on each test image and show the results:

def show_inference(model, image):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
##  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  display(Image.fromarray(image_np))

while True:
  screen = cv2.resize(grab_screen(region=(0,40,1280,768)), (160,120))
  show_inference(detection_model, screen)
  cv2.imshow('window', screen)
  if cv2.waitKey(25) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    break



### ## Instance Segmentation
##
##model_name = "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
##masking_model = load_model(model_name)
##
##
### The instance segmentation model includes a `detection_masks` output:
##
##masking_model.output_shapes
##
##
##for image_path in TEST_IMAGE_PATHS:
##  show_inference(masking_model, image_path)
##

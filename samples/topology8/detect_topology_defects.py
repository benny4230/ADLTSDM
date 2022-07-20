import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import json
import pickle
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.config import Config
from mrcnn.model import log

############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Number of Classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "/home/benny/Documents/Mask_RCNN-master/samples/topology8/mask_rcnn_object_0149.h5")


# Directory of images to run detection on
#IMAGE_DIR = os.path.join("/home/benny/Documents/Mask_RCNN-master/datasets/dpcrexp2/val")
#IMAGE_DIR = os.path.join("/home/benny/Documents/Mask_RCNN-master/datasets/dpcr_40_version3")
IMAGE_DIR = os.path.join("/home/benny/Documents/Mask_RCNN-master/datasets/topology8cut/val")
#IMAGE_DIR = os.path.join("/home/benny/Documents/Mask_RCNN-master/datasets/dpcr_40_version2/val")

class InferenceConfig(CustomConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the bottle dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "defect")


        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations1 = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        # print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Keep track of file names in annotation for use later
        filenames = []
        for f in annotations1:
            filenames.append(f)

        # Add images
        i = 0
        for a in annotations:
            #print([r['shape_attributes'] for r in a['regions'].values()])
            #print([s['region_attributes'] for s in a['regions'].values()])
            #print(a['regions'].values())
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['category_id'] for s in a['regions']]
            #print("objects:",objects)

            # Follow and map according to the category id: value
            name_dict = {"defect": 1}
            # key = tuple(name_dict)
            num_ids = [name_dict[a] for a in objects]

            # num_ids = [int(n['Event']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            #print("numids",num_ids)
            print(a['filename'])
            #print(a)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)

            i = i + 1

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a bottle dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
        	rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

        	mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=15,
                layers='all')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    fig.tight_layout()
    return ax

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("successfully save the txt file!")
############################################################
#  Inference
############################################################

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'dpcr']

# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]
#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))


# Load Validation dataset
dataset = CustomDataset()
#dataset.load_custom("/home/benny/Documents/Mask_RCNN-master/datasets/dpcr_40_version2/", "val")
dataset.load_custom("/home/benny/Documents/Mask_RCNN-master/datasets/topology8cut/", "val")
#dataset.load_custom("/home/benny/Documents/Mask_RCNN-master/datasets/dpcrexp2/", "val")
#dataset.load_custom("/home/benny/Documents/Mask_RCNN-master/datasets/dpcr_40/", "val")
dataset.prepare()

counter = 0
for image_id in dataset.image_ids:
    print("Image ID: "+str(image_id))
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    #info = dataset.image_info[image_id]
    #print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
    #                                   dataset.image_reference(image_id)))
    #print("Original image shape: ", modellib.parse_image_meta(image_meta[np.newaxis,...])["original_image_shape"][0])

    if counter == 0:
        save_bbox, save_class_id, save_mask = gt_bbox, gt_class_id, gt_mask
    else:
        save_bbox = np.concatenate((save_bbox,gt_bbox), axis=0)
        save_class_id = np.concatenate((save_class_id,gt_class_id), axis=0)
        save_mask = np.concatenate((save_mask,gt_mask), axis=2)

    # Run detection
    results = model.detect([image], verbose=1)
    r = results[0]

    # Visualize results
    #print(r)
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                        class_names, r['scores'])

    if counter == 0:
        save_rois_detect, save_ids_detect, save_score_detect, save_masks_detect =\
            r['rois'],  r['class_ids'], r['scores'], r['masks']
    else:
        save_rois_detect = np.concatenate((save_rois_detect,r['rois']), axis=0)
        save_ids_detect = np.concatenate((save_ids_detect,r['class_ids']), axis=0)
        save_score_detect = np.concatenate((save_score_detect,r['scores']), axis=0)
        save_masks_detect = np.concatenate((save_masks_detect,r['masks']), axis=2)
    counter += 1

# Compute AP
AP, precisions, recalls, overlaps =\
    utils.compute_ap(save_bbox, save_class_id, save_mask, save_rois_detect,
        save_ids_detect, save_score_detect, save_masks_detect,iou_threshold=0.1)

#save precision and recall txt file
text_save('precision_all.txt', precisions)
text_save('recall_all.txt', recalls)

print("AP: ", AP)
print("mAP: ", np.mean(AP))

#Obtain yolov3 precision and recall
#fr = open('ddPCR_pr.pkl','rb')#the route of the pkl file
#inf = pickle.load(fr)
#fr.close()
#x=inf['rec']
#y=inf['prec']
#right version

plt.plot(recalls, precisions, 'b', label='Mask RCNN PR Curve') #mask rcnn pr-curve
#plt.plot(x, y, 'r', label='Yolov3 PR Curve')   #yolov3 pr-curve\
#plt.plot(x2, y2, 'g', label='Improved Yolov3 PR Curve') #improved yolov3 curve
#plt.plot(x3, y3, 'tan', label='Vanilla Yolov3 without RBTM & STAM') #improved yolov3 curve
plt.title('precision-recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim(0.5, 1.05)
plt.ylim(0.5, 1.05)
plt.legend()
plt.savefig('PR_Curve.png')
plt.show()







###########################################################
#  Evaluation
############################################################


# Compute VOC-Style mAP @ IoU=0.5
# Running on N images. Increase for better accuracy.
#image_ids = np.random.choice(dataset_val.image_ids, 200)
#APs = []
#for image_id in image_ids:
#    # Load image and ground truth data
#    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#        modellib.load_image_gt(dataset_val, config,
#                               image_id, use_mini_mask=False)
#    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
#    # Run object detection
#    results = model.detect([image], verbose=0)
#    r = results[0]
#    # Compute AP
#    AP, precisions, recalls, overlaps =\
#        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#                         r["rois"], r["class_ids"], r["scores"], r['masks'])
#    APs.append(AP)
#
#print("mAP: ", np.mean(APs))

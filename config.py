import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
import warnings

warnings.filterwarnings('ignore')

DATASET = r"D:\ML_Projects\YOLOv3_Custom_Implementation\Data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 #10
BATCH_SIZE = 32
IMAGE_SIZE = 224 #416
NUM_CLASSES = 2 #20
LEARNING_RATE = 1e-4 #1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 1024 #256 #32
CONF_THRESHOLD = 0.5 #0.5
MAP_IOU_THRESH = 0.55 #0.5
NMS_IOU_THRESH = 0.55 #0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
SAVE_MODEL = True
SAVE_MODEL_NAME = r"D:\ML_Projects\YOLOv3_Custom_Implementation\Models\fmd_yolov3_14.pth.tar"
LOAD_MODEL = True
LOAD_MODEL_NAME = r"D:\ML_Projects\YOLOv3_Custom_Implementation\Models\fmd_yolov3_12.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"
PRINT_METRIC = NUM_EPOCHS / 4 # The denominator times the evaluation metrics will print
SAVE_CHECKPOINT_FREQ = NUM_EPOCHS / 16 # The denominator times the model checkpoint will save
TRAIN_FILE = "/annotations.csv"
TEST_FILE = "/annotations.csv"
TRAIN_IMG_NAMES = DATASET + "/image names.csv"
TEST_IMG_NAMES = DATASET + "/image names.csv"
EVAL_IMG_NAMES = DATASET + "/image names.csv"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.Affine(shear=15, p=0.5, mode=cv2.BORDER_CONSTANT),
            ],
            p=0.1, #1.0,
        ),
        A.HorizontalFlip(p=0.1),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

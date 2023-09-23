import albumentations as A
from albumentations.pytorch import ToTensorV2

# mobilenet
# m=(0.485, 0.456, 0.406)
# std=(0.229, 0.224, 0.225)

# efficientnet lite
m = (0.498, 0.498, 0.498)
std = (0.502, 0.502, 0.502)

class CustomCoarseDropout(A.CoarseDropout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def apply_to_keypoints(
        self, keypoints, holes, **params):
        result = set(keypoints)
        return list(result)

keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)

augment = A.Compose([
    A.Perspective(p=0.2),
    A.ShiftScaleRotate (rotate_limit = 180, p=0.3),
    A.Resize(224,224),
    A.HorizontalFlip(p=0.3),
    A.RandomSunFlare(src_radius=30,p=0.3),
    # A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.4),
    CustomCoarseDropout(p=0.5, min_height=8, min_width=8, max_height=30, max_width=30, max_holes=6),
    A.OneOf([
            A.RGBShift(p=0.4),
            A.ChannelShuffle(p=0.4),
            A.HueSaturationValue(p=0.6),
            A.RGBShift(p=0.4)
        ], p=0.4),
    A.RandomBrightnessContrast(p=0.3, brightness_limit=(0,0.2), contrast_limit=(0,0.2)),
    A.AdvancedBlur(p=0.3),
    # A.Equalize(p=1),
    A.Normalize(mean=m, std=std),
    ToTensorV2(),
], keypoint_params=keypoint_params)

normal_transform = A.Compose([
    A.Resize(224,224),
    # A.Equalize(p=1),
    A.Normalize(mean=m, std=std),
    ToTensorV2(),
], keypoint_params=keypoint_params)
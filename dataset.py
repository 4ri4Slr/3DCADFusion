import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import matplotlib.pyplot as plt


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def test_visualize(image, mask, res):

    image = cv2.resize(image, (res[0], res[1]))
    mask = cv2.resize(mask, (res[0], res[1]))

    plt.figure()
    plt.imshow(image, 'gray', interpolation='none')
    plt.imshow(mask, 'ocean', interpolation='none', alpha=0.4)
    plt.show()



from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):

    CLASSES = ['unlabeled', 'hand']

    def __init__(
            self,
            images_dir,
            masks_dir = None,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):

        self.ids = sorted(os.listdir(images_dir))
        self.ids.sort(key=lambda x: int(os.path.splitext(x)[0]))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        if masks_dir:
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mask_dir = masks_dir
    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.rotate(image, cv2.cv2.ROTATE_180)
        lum =  0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
        image = np.stack((lum,) * 3, axis=-1).astype('uint8')

        if self.mask_dir:
            mask = cv2.imread(os.path.splitext(self.masks_fps[i])[0]+'.png',0)
            mask[mask != 0] = 1
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')

            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

            return image, mask
        else:

            if self.augmentation:
                sample = self.augmentation(image=image)
                image = sample['image']

            if self.preprocessing:

                sample = self.preprocessing(image=image)
                image = sample['image']

            return image


    def __len__(self):
        return len(self.ids)


import albumentations as albu


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p = 0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""

    h = 400
    w = 500

    test_transform = [
        #albu.CenterCrop(2000, 2500),

        albu.Resize(h, w),
        albu.PadIfNeeded(int((np.floor(h/32)+1) * 32), int((np.floor(w/32)+1) * 32))
        #albu.Resize(320,320)
    ]
    return albu.Compose(test_transform)


def get_visualization_augmentation():
    """Add paddings to make image shape divisible by 32"""

    test_transform = [

        #albu.CenterCrop(2000, 2500),
        albu.Resize(400, 500)
    ]
    return albu.Compose(test_transform)




def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


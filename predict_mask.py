import torch
import dataset
import os
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['unlabeled', 'hand']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


def predict(path,res):

    best_model = torch.load('./best_model.pth')
    test_dataset = dataset.Dataset(
        os.path.join(path, 'rectified_rgb'),
        augmentation = dataset.get_validation_augmentation(),
        preprocessing= dataset.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    test_dataset_vis = dataset.Dataset(
        os.path.join(path, 'rectified_rgb'),
        augmentation= dataset.get_visualization_augmentation(),
        classes=CLASSES,
    )

    for n in range((len(test_dataset))):

        image_vis = test_dataset_vis[n].astype('uint8')
        image = test_dataset[n]
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy()>0.0001)*1
        pr_mask = pr_mask.astype('uint8')

        im_width, im_height = image_vis.shape[:2]
        mask_width, mask_height = pr_mask.shape
        cropped_mask = pr_mask[(mask_width-im_width)//2:(mask_width-im_width)//2+im_width, (mask_height-im_height)//2:(mask_height-im_height)//2+im_height]
        cropped_mask2 = cv2.resize(cropped_mask, (res[0], res[1]))
        #cropped_mask2 = cv2.rotate(cropped_mask2, cv2.cv2.ROTATE_180)
        cv2.imwrite(os.path.join(path, 'masks frame-' + str(n).zfill(6) + '.color.jpg'), cropped_mask2 )
        masked = np.ma.masked_where(cropped_mask == 0, cropped_mask)
        print(n)
        # plt.figure()
        # plt.imshow(image_vis, 'gray', interpolation='none')
        # plt.imshow(masked, 'ocean', interpolation='none', alpha=0.6)
        # plt.show()
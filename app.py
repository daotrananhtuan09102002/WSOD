import streamlit as st
import os
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import keras_cv
import cv2
import io

from PIL import Image
from val import get_model
from torch.utils.data import Dataset, DataLoader
from data_loaders import _IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE
from wsod.util import get_prediction

st.set_page_config(page_title="Object Detection", page_icon=":camera:", layout="wide")


class_ids = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    "None"
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))



class NetDataset(Dataset):
    def __init__(self, raw_images, labels, transform, num_classes=20):
        self.raw_images = raw_images
        self.num_classes = num_classes
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, index):
        raw_image = self.raw_images[index]

        image = self.transform(raw_image)

        # Convert class indices to a multi-hot matrix
        if None not in self.labels[index]:
            label_multi_hot = F.one_hot(
                torch.unique(torch.tensor(self.labels[index])),
                num_classes=self.num_classes
            ).sum(dim=0)
            return image, label_multi_hot.float()

        return image, None

def load_image(image_file):
    img = Image.open(image_file)
    return img

def upload_image(img_file):
    # print query image and cropped image
    if img_file:
        img = Image.open(img_file)
        st.header("Uploaded image")
        st.image(img, use_column_width=True)
        return img
    return None


if __name__ == "__main__":
    st.sidebar.subheader('Step 1: Select image to detect')
    img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
    st.sidebar.subheader('Step 2: Select type of model to use')
    type_model = st.sidebar.selectbox("Type of model", ["ResNet50", "VGG16"], label_visibility="collapsed")
    st.sidebar.subheader('Step 3: Select classes to detect')
    classes = st.sidebar.multiselect("Select classes", class_ids, default=None, placeholder="None")
    st.sidebar.subheader('Step 4: Select cam threshold')
    cam_threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=1.0, step=0.05, value=0.4)

    img = upload_image(img_file)
    button = st.sidebar.button('Detect')

    if button:
        parser = argparse.ArgumentParser(description='Image Retrieval')
        parser.add_argument('--dataset_name', type=str, default='VOC', help='Dataset name')
        parser.add_argument('--architecture', type=str, default='vgg16', help='Model architecture')
        parser.add_argument('--architecture_type', type=str, default='drop', help='Model architecture type')
        parser.add_argument('--pretrained', type=bool, default=True, help='Use pre-trained weights')
        parser.add_argument('--large_feature_map', action='store_true', help='Use large feature map')
        parser.add_argument('--drop_threshold', type=float, default=0.8, help='Drop threshold')
        parser.add_argument('--drop_prob', type=float, default=0.25, help='Drop probability')
        parser.add_argument('--resize_size', type=int, default=224, help='Resize size')
        parser.add_argument('--checkpoint_path', required=True, type=str, default=None, help='Checkpoint path')
        
        if type_model == "ResNet50":
            args = parser.parse_args('--checkpoint_path /content/drive/MyDrive/WSOD_Torch/weights_torch_VOC-2012/resnet_drop_APL_2012_small.pth.tar \
                                     --architecture resnet50 \
                                    --architecture_type drop \
                                    --drop_threshold 0.8 \
                                    --drop_prob 0.25'.split())
        else:
            args = parser.parse_args('--checkpoint_path /content/drive/MyDrive/WSOD_Torch/weights_torch_VOC-2012/vgg_cam_BCE_2012.pth.tar \
                                     --architecture vgg16 \
                                     --architecture_type cam \
                                     --drop_threshold 0.8 \
                                     --drop_prob 0.5'.split())

        tf = transforms.Compose([
            transforms.Resize((args.resize_size, args.resize_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE),
        ])

        x = img
        y = [class_ids.index(c) for c in classes] if classes is not None else None

        model = get_model(args)

        checkpoint_path = args.checkpoint_path
        if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                print("Check {} loaded.".format(checkpoint_path))
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        model.eval()

        with torch.no_grad():
            if classes is None:
                y_pred = model(x.cuda(), return_cam=True)
            else:
                y_pred = model(x.cuda(), labels=y, return_cam=True)

        for img_idx in range(x.shape[0]):
            orig_img = x[img_idx] * torch.tensor([.229, .224, .225]).view(3, 1, 1) + torch.tensor([0.485, .456, .406]).view(3, 1, 1)
            orig_img = orig_img.numpy().transpose([1, 2, 0])
            orig_img = cv2.normalize(orig_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)

            for gt_class, cam in y_pred['cams'][img_idx].items():
                fig, axs = plt.subplots(2, 3, figsize=(12, 6), num=1, clear=True, layout="constrained")
                cam = cam.cpu().numpy()

                # row 1 col 1: Original Image
                axs[0, 0].imshow(orig_img)
                axs[0, 0].set_title(f'{class_mapping[gt_class]}')
                axs[0, 0].axis('off')

                # row 2 col 1: Colorbar with 'hot' cmap
                heatmap = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
                heatmap = cv2.normalize(heatmap, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

                im2 = axs[1, 0].imshow(heatmap, cmap='Reds')
                axs[1, 0].set_title('CAM')
                axs[1, 0].axis('off')
                cbar2 = plt.colorbar(im2, ax=axs[1, 0])

                # row 1 col 2: Thresholded CAM
                heatmap = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
                heatmap = cv2.normalize(heatmap, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)

                _, thr_gray_heatmap = cv2.threshold(
                    src=heatmap,
                    thresh=cam_threshold * np.max(heatmap),
                    maxval=255,
                    type=cv2.THRESH_BINARY
                )
                axs[0, 1].imshow(thr_gray_heatmap, cmap='gray')
                axs[0, 1].set_title(f'Thresholded @{cam_threshold:.2f}')
                axs[0, 1].axis('off')

                # row 1 col 4: Bbox
                pred = get_prediction([{gt_class: y_pred['cams'][img_idx][gt_class]}], cam_threshold=cam_threshold)[0].numpy()

                temp_fig = keras_cv.visualization.plot_bounding_box_gallery(
                    [orig_img],
                    value_range=(0, 255),
                    rows=1,
                    cols=1,
                    y_pred={'classes': [pred[:, 4]], 'boxes': [pred[:, :4]]},
                    scale=5,
                    font_scale=0.7,
                    bounding_box_format='xyxy',
                    line_thickness=1,
                    show=None,
                    path=None
                )

                buffer = io.BytesIO()
                temp_fig.savefig(buffer, format='PNG')
                plt.close(temp_fig)
                axs[1, 1].imshow(np.asarray(Image.open(buffer)))
                axs[1, 1].axis('off')
                axs[1, 1].set_title('Prediction')

                ksizes = [7]
                for idx, ksize in enumerate(ksizes, start=2):
                    # heatmap
                    heatmap = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
                    heatmap = cv2.normalize(heatmap, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
                    heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), 0)

                    _, thr_gray_heatmap = cv2.threshold(
                        src=heatmap,
                        thresh=0,
                        maxval=255,
                        type=cv2.THRESH_BINARY+cv2.THRESH_OTSU
                    )
                    axs[0, idx].imshow(thr_gray_heatmap, cmap='gray')
                    axs[0, idx].set_title(f'Otsu with\nGauss blur ksize={ksize}')
                    axs[0, idx].axis('off')

                    # Bbox
                    pred = get_prediction([{gt_class: y_pred['cams'][img_idx][gt_class]}], cam_threshold=None, gaussian_ksize=ksize)[0].numpy()

                    temp_fig = keras_cv.visualization.plot_bounding_box_gallery(
                        [orig_img],
                        value_range=(0, 255),
                        rows=1,
                        cols=1,
                        y_pred={'classes': [pred[:, 4]], 'boxes': [pred[:, :4]]},
                        scale=5,
                        font_scale=0.7,
                        bounding_box_format='xyxy',
                        line_thickness=1,
                        show=None,
                        path=None
                    )

                    buffer = io.BytesIO()
                    temp_fig.savefig(buffer, format='PNG')
                    plt.close(temp_fig)
                    axs[1, idx].imshow(np.asarray(Image.open(buffer)))
                    axs[1, idx].axis('off')
                    axs[1, idx].set_title('Prediction')
                
                st.pyplot(fig)
                plt.show()





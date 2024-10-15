import torch
from torch.utils.data import Dataset
import os
import glob
import cv2
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import ToPILImage
import dill as pickle
from xml.etree import ElementTree as ET

def transform_data():
    def get_train_transform():
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            T.ToTensor(),
        ])

    def get_valid_transform():
        return T.Compose([
            T.ToTensor(),
        ])

    def is_valid_xml(file_path):
        try:
            tree = ET.parse(file_path)
            return True
        except ET.ParseError as e:
            print(f"XML ParseError for {file_path}: {e}")
            return False

    class CustomDataset(Dataset):
        def __init__(self, images_path, labels_path, directory, width, height, classes, transforms=None):
            self.transforms = transforms
            self.images_path = images_path
            self.labels_path = labels_path
            self.directory = directory
            self.height = height
            self.width = width
            self.classes = classes
            self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
            self.all_image_paths = []

            for file_type in self.image_file_types:
                self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
            self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
            self.all_images = sorted(self.all_images)
            print("Number of images: ", len(self.all_images))

        def load_img(self, img_path):
            print("img_path-------------------", img_path)
            image = cv2.imread(img_path)

            if image is None:
                raise ValueError(f"Image not found or unable to load: {img_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return image

        def normalize_bbox(self, bboxes, rows, cols):
            norm_bboxes = np.zeros_like(bboxes)
            if cols > 0 and rows > 0:
                norm_bboxes[:, 0] = bboxes[:, 0] / cols
                norm_bboxes[:, 1] = bboxes[:, 1] / rows
                norm_bboxes[:, 2] = bboxes[:, 2] / cols
                norm_bboxes[:, 3] = bboxes[:, 3] / rows
            return norm_bboxes

        def __getitem__(self, index):
        
        
            try:
                image_id = self.all_images[index][:-4]
                annot_file_path = os.path.join(self.labels_path, f"{image_id}.xml")

                # Validate XML before parsing
                if not is_valid_xml(annot_file_path):
                    print(f"Skipping corrupted or empty XML: {annot_file_path}")
                    return None  # You can return a default image and empty target here if necessary

                # Parse the annotation file
                tree = ET.parse(annot_file_path)
                root = tree.getroot()

                image_path = os.path.join(self.images_path, self.all_images[index])

                # Attempt to load the image
                image = self.load_img(image_path)

            except ValueError as e:
                print(f"Error loading image: {e}")
                return None  # Skip this image

            boxes = []
            labels = []
            for member in root.findall('object'):
                label = member.find('Target').text
                if label not in self.classes:
                    continue  # Skip labels not in the classes
                labels.append(self.classes.index(label))
                x_center = float(member.find('x').text)
                y_center = float(member.find('y').text)
                width = float(member.find('width').text)
                width = x_center + width
                height = float(member.find('height').text)
                height = y_center + height

                if not any(np.isnan([x_center, y_center, width, height])) and width > 0 and height > 0:
                    boxes.append([x_center, y_center, width, height])

            boxes = np.array(boxes) if boxes else np.empty((0, 4))  # Use empty array if no valid boxes
            area = boxes[:, 2] * boxes[:, 3] if boxes.size > 0 else torch.tensor([0], dtype=torch.float32)
            area = torch.as_tensor(area, dtype=torch.float32)

            labels = torch.tensor(labels, dtype=torch.long) if labels else torch.tensor([], dtype=torch.long)

            image = (image * 255).astype(np.uint8)

            image_pil = ToPILImage()(image)

            if self.transforms:
                image = self.transforms(image_pil)

            _, h, w = image.shape
            norm_boxes = self.normalize_bbox(boxes, rows=h, cols=w)

            valid_indices = (norm_boxes[:, 2] > 0) & (norm_boxes[:, 3] > 0)
            valid_boxes = norm_boxes[valid_indices]

            target = {}
            if valid_boxes.size > 0:
                target['boxes'] = torch.as_tensor(valid_boxes, dtype=torch.float32)
                target['labels'] = labels[valid_indices]
            else:
                target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                target['labels'] = torch.empty((0,), dtype=torch.long)

            target['image_id'] = torch.tensor([index])
            target['area'] = area[valid_indices] if valid_indices.any() else torch.tensor([0], dtype=torch.float32)

            return image, target

        def __len__(self):
            return len(self.all_images)

    IMAGE_WIDTH = 800
    IMAGE_HEIGHT = 680
    classes = ['0', '1']

    train_dataset = CustomDataset(
        os.path.join(os.getcwd(), "output_images"),
        os.path.join(os.getcwd(), "xml_labels"),
        os.getcwd(),
        IMAGE_WIDTH, IMAGE_HEIGHT,
        classes,
        get_train_transform()
    )
    print("Train Dataset: ", train_dataset)

    valid_dataset = CustomDataset(
        os.path.join(os.getcwd(), "output_images"),
        os.path.join(os.getcwd(), "xml_labels"),
        os.getcwd(),
        IMAGE_WIDTH, IMAGE_HEIGHT,
        classes,
        get_valid_transform()
    )
    print("Valid Dataset: ", valid_dataset)

    i, a = train_dataset[1347]
    print("Image: ", i)
    print("Annotations: ", a)

    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)

    with open('valid_dataset.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)

    return train_dataset

transform_data()

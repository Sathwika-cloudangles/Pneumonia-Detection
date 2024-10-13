import torch
from torch.utils.data import Dataset
import os
from xml.etree import ElementTree as ET
import glob as glob
import cv2
import numpy as np
import random
from torchvision import transforms as T
from torchvision.transforms import ToPILImage
import dill as pickle

def transform_data():
    # Define the training transforms
    def get_train_transform():
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            T.ToTensor(),  # Ensure the output is a tensor
        ])

    def get_valid_transform():
        return T.Compose([
            T.ToTensor(),  # Convert the image to a tensor
        ])

    class CustomDataset(Dataset):
        def __init__(self, images_path, labels_path, labels_txt, directory,
                     width, height, classes, transforms=None, 
                     use_train_aug=False,
                     train=False, mosaic=False):
            self.transforms = transforms
            self.use_train_aug = use_train_aug
            self.images_path = images_path
            self.labels_path = labels_path
            self.directory = directory
            self.labels_txt = labels_txt
            self.height = height
            self.width = width
            self.classes = classes
            self.train = train
            self.mosaic = mosaic
            self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
            self.all_image_paths = []
            
            # Get all the image paths in sorted order
            for file_type in self.image_file_types:
                self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
            self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))
            self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
            self.all_images = sorted(self.all_images)
            print("Number of images:-----------------", len(self.all_images))

        def load_image_and_labels(self, index):
            if index >= len(self.all_images):
                raise IndexError("Index out of range")
            image_name = self.all_images[index]
            image_path = os.path.join(self.images_path, image_name)

            # Read the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image_resized = cv2.resize(image, (self.width, self.height))
            image_resized /= 255.0

            # Capture the corresponding XML file for getting the annotations
            annot_filename = image_name[:-4] + '.xml'
            annot_file_path = os.path.join(self.labels_path, annot_filename)

            boxes = []
            orig_boxes = []
            labels = []
            tree = ET.parse(annot_file_path)
            root = tree.getroot()

            # Get the height and width of the image
            image_width = image.shape[1]
            image_height = image.shape[0]

            # Box coordinates for xml files are extracted and corrected for image size
            for member in root.findall('object'):
                labels.append(self.classes.index(member.find('Target').text))
                x_center = float(member.find('x').text)
                y_center = float(member.find('y').text)
                width = float(member.find('width').text)
                height = float(member.find('height').text)

                xmin = (x_center - width / 2) * image_width
                ymin = (y_center - height / 2) * image_height
                xmax = (x_center + width / 2) * image_width
                ymax = (y_center + height / 2) * image_height

                # Ensure xmax and ymax are not greater than the image dimensions
                ymax, xmax = self.check_image_and_annotation(xmax, ymax, image_width, image_height)

                # Filter out invalid boxes (zeros or NaN)
                if not (xmax <= xmin or ymax <= ymin or np.isnan(xmin) or np.isnan(ymin) or np.isnan(xmax) or np.isnan(ymax)):
                    orig_boxes.append([xmin, ymin, xmax, ymax])
                    xmin_final = (xmin/image_width) * self.width
                    xmax_final = (xmax/image_width) * self.width
                    ymin_final = (ymin/image_height) * self.height
                    ymax_final = (ymax/image_height) * self.height
                    boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

            # Handle zero-length boxes and NaN
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            if len(boxes) == 0:
                return image, image_resized, [], [], [], None, None, (image_width, image_height)
            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.as_tensor([], dtype=torch.float32)
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            return image, image_resized, orig_boxes, boxes, labels, area, iscrowd, (image_width, image_height)

        def check_image_and_annotation(self, xmax, ymax, width, height):
            """Check that all x_max and y_max are not more than the image width or height."""
            if ymax > height:
                ymax = height
            if xmax > width:
                xmax = width
            return ymax, xmax

        def __getitem__(self, idx):
            print("Index-----------------------", idx)
            # Load image and labels
            if not self.mosaic:
                image, image_resized, orig_boxes, boxes, \
                    labels, area, iscrowd, dims = self.load_image_and_labels(idx)

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # Prepare the final `target` dictionary
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd
            image_id = torch.tensor([idx])
            target["image_id"] = image_id

            # Convert to PIL Image before applying transforms
            image_resized = ToPILImage()(image_resized)  # Convert NumPy array to PIL Image

            # Apply transformations directly to the PIL image
            if self.use_train_aug:  # Use train augmentation if argument is passed.
                image_resized = self.transforms(image_resized)
            else:
                image_resized = self.transforms(image_resized)

            # Return image as tensor directly from transforms
            return image_resized, target


        def __len__(self):
            return len(self.all_images)

    IMAGE_WIDTH = 800
    IMAGE_HEIGHT = 680
    classes = ['0', '1']
    
    # Create datasets
    train_dataset = CustomDataset(
        os.path.join(os.getcwd(), "output_images"),
        os.path.join(os.getcwd(), "xml_labels"), 
        os.path.join(os.getcwd(), "txt_labels"),
        "Pnemonia",
        IMAGE_WIDTH, IMAGE_HEIGHT, 
        classes, 
        get_train_transform()
    )
    print("Train Dataset: ", train_dataset)
    
    valid_dataset = CustomDataset(
        os.path.join(os.getcwd(), "output_images"),
        os.path.join(os.getcwd(), "xml_labels"), 
        os.path.join(os.getcwd(), "txt_labels"),
        "Pnemonia",
        IMAGE_WIDTH, IMAGE_HEIGHT, 
        classes, 
        get_valid_transform()
    )
    print("Valid Dataset: ", valid_dataset)
    
    i, a = train_dataset[15]
    print("Image: ", i)
    print("Annotations: ", a)
    
    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    
    with open('valid_dataset.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)

    return train_dataset 

transform_data()

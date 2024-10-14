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
            orig_height, orig_width, _ = image.shape

            # Resize the image
            image_resized = cv2.resize(image, (self.width, self.height))
            image_resized /= 255.0
            
            print(image_resized)

            # Capture the corresponding XML file for getting the annotations
            annot_filename = image_name[:-4] + '.xml'
            annot_file_path = os.path.join(self.labels_path, annot_filename)

            boxes = []
            labels = []

            tree = ET.parse(annot_file_path)
            root = tree.getroot()

            # Box coordinates for xml files are extracted and corrected for image size
            for member in root.findall('object'):
                labels.append(self.classes.index(member.find('Target').text))
                x_center = float(member.find('x').text)
                y_center = float(member.find('y').text)
                width = float(member.find('width').text)
                height = float(member.find('height').text)

                # Original bounding box coordinates in terms of the original image
                xmin = (x_center - width / 2) * orig_width
                ymin = (y_center - height / 2) * orig_height
                xmax = (x_center + width / 2) * orig_width
                ymax = (y_center + height / 2) * orig_height

                # Ensure coordinates are within image bounds
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(orig_width, xmax)
                ymax = min(orig_height, ymax)

                if xmax <= xmin or ymax <= ymin:
                    continue  # Skip invalid boxes

                # Now scale the bounding box to match the resized image dimensions
                xmin_resized = (xmin / orig_width) * self.width
                ymin_resized = (ymin / orig_height) * self.height
                xmax_resized = (xmax / orig_width) * self.width
                ymax_resized = (ymax / orig_height) * self.height

                xmin_resized = max(0, min(self.width, xmin_resized))
                ymin_resized = max(0, min(self.height, ymin_resized))
                xmax_resized = max(0, min(self.width, xmax_resized))
                ymax_resized = max(0, min(self.height, ymax_resized))
                
                # Add the resized bounding box to the list
                boxes.append([xmin_resized, ymin_resized, xmax_resized, ymax_resized])

            # Convert boxes and labels to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            if boxes.shape[0] == 0:
                return image, image_resized, [], [], [], None, None, (orig_width, orig_height)

            # Calculate area and other target data
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            return image, image_resized, boxes, labels, area, iscrowd, (orig_width, orig_height)


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
                image, image_resized, boxes, \
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

            image_resized = (image_resized * 255).astype(np.uint8)  # Ensure the values are within uint8 range
            image_resized = ToPILImage()(image_resized)  # Convert to PIL image

            # Apply transformations directly to the PIL image
            if self.use_train_aug:  # Use train augmentation if argument is passed.
                image_resized = self.transforms(image_resized)
            else:
                image_resized = self.transforms(image_resized)

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
    
    i, a = train_dataset[1347]
    print("Image: ", i)
    print("Annotations: ", a)
    
    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    
    with open('valid_dataset.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)

    return train_dataset 

transform_data()

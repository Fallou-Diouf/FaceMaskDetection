import os
import xml.etree.ElementTree as ET

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image


classes_index = {
    'with_mask': 1,
    'without_mask': 2,
    'mask_weared_incorrect': 3
}


def get_objects(xml_file):
    annotation = ET.parse(xml_file)
    root = annotation.getroot()

    objects = []

    for obj in root.findall('object'):
        new_object = {'name': obj.find('name').text}

        bbox = obj.find('bndbox')

        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        new_object['bbox'] = [xmin, ymin, xmax, ymax]
        objects.append(new_object)

    return objects


class FaceMaskDataset(Dataset):

    def __init__(
        self,
        img_folder,
        annotation_folder,
        indexes,
        conversion=T.ToTensor()
    ):
        self.conversion = conversion

        self.dataset = []

        for index in tqdm(indexes):

            sample = {}

            sample['image'] = Image.open(
                os.path.join(
                    img_folder,
                    'images',
                    'maksssksksss' + str(index) + '.png'
                )
            ).convert('RGB')

            sample['objects'] = get_objects(
                os.path.join(
                    annotation_folder,
                    'annotations',
                    'maksssksksss' + str(index) + '.xml'
                )
            )

            sample['id'] = index

            self.dataset.append(sample)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        target = {
            'boxes': [],
            'labels': []
        }

        for obj in self.dataset[idx]['objects']:
            target['boxes'].append(obj['bbox'])
            target['labels'].append(classes_index[obj['name']])

        target['boxes'] = torch.as_tensor(
            target['boxes'],
            dtype=torch.float32
        )

        target['labels'] = torch.as_tensor(
            target['labels'],
            dtype=torch.int64
        )

        target['image_id'] = torch.tensor(
            [self.dataset[idx]['id']
        ])

        img = self.dataset[idx]['image']

        if self.conversion is not None:
            img = self.conversion(img)

        return img, target
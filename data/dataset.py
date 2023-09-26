import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning as pl
import cv2 as cv
from tqdm import tqdm
import json
from data.augmentation import augment, normal_transform

char2index = {
    "poster": 0,
    "page": 1,
    "opened_book": 2,
    "cover": 3,
    "card": 4,
    "document": 5
}

def json_load(path):
    with open(path, "r") as f:
        return json.load(f)

def scale(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
def rearrange_points(l):
    l = sorted(l, key = lambda c: c[1])
    l[:2] = sorted(l[:2], key = lambda c: c[0])
    l[2:4] = sorted(l[2:4], key = lambda c: c[0], reverse = True)
    
    output = [scale(x, -260, 520, -1, 1) for sublist in l for x in sublist]
    # output = [x for sublist in l for x in sublist]
    
    return output    

class DocDataModule(pl.LightningDataModule):
    def __init__(self, train_json_path, valid_json_path, data_dir, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_json_data = json_load(train_json_path)
        self.valid_json_data = json_load(valid_json_path)
        self.data_dir = data_dir
    
    def setup(self, stage):
        self.train_dataset = DocDataset(self.train_json_data, self.data_dir, transform = augment)
        self.valid_dataset = DocDataset(self.valid_json_data, self.data_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False
        )

class DocDataset(Dataset):
    def __init__(self, data, data_dir, transform = normal_transform):
        super().__init__()

        self.data = data
        print(len(self.data))
        self.data_dir = data_dir
        self.transform = transform
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # image = self.load_image(self.data_list[index]['image_path'])
        # mask = self.load_image(self.data_list[index]['mask_path'])

        image, corners = self.load_image(self.data[index]['image_path'], self.data[index]['corners'])
        cls = torch.tensor(char2index[self.data[index]["class"]])

        return image, corners, cls
    
    def load_image(self, path, corners):
        path = self.data_dir +"/"+ path
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # image = transforms.ToTensor()(image)
        # image = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image)
        
        transformed = self.transform(image=image, keypoints = corners)
        transformed_image = transformed["image"]
        transformed_corners = rearrange_points(transformed["keypoints"])
        
        return transformed_image, torch.tensor(transformed_corners, dtype = torch.float)
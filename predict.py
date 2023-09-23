import torch
import os
from models.model import LDRNet
import lightning as pl
import torchvision.transforms as transforms
import cv2 as cv
import configs
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data.augmentation import normal_transform

def load_image(image_path):
    path = image_path
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # image = cv.resize(image, (224,224))
    # image = transforms.ToTensor()(image)
    # image = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image)
    transformed = normal_transform(image=image, keypoints = [(0,0)])
    transformed_image = transformed["image"]
    return transformed_image, transformed_image.shape

model = LDRNet(configs.n_points, lr = configs.lr)
# model = model.load_from_checkpoint("/notebooks/LDRNet/all/epoch=164-step=49995.ckpt")
model = model.load_from_checkpoint("efficientnet_all/epoch=200-step=60903.ckpt")
model.eval()

input_dir = "15_images_easy"
output_dir = "output_images"
for f in tqdm(os.listdir(input_dir)):
    if not (f.endswith(".jpg") or f.endswith(".jpeg")):
        continue
    image_path = input_dir + "/" + f
    print(image_path)

    image, _ = load_image(image_path)
    image = image.cuda()
    corners, points = model(image.unsqueeze(0))
    
    output_image_path = output_dir + "/" + f
    
    img = cv.imread(image_path)
    img = cv.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)))
    corners = corners[0].detach().cpu().numpy()
    points = points[0].detach().cpu().numpy()
    x = corners[0::2] * img.shape[1]
    y = corners[1::2] * img.shape[0]
    
    colors = [(0,0,255), (0,255,0), (255,0,0), (255,0,255)]
    for i in range(0,4):
        # s = (int(corners[i % 4][0]*img.shape[1]), int(corners[i % 4][1]*img.shape[0]))
        # e = (int(corners[(i+1) % 4][0]*img.shape[1]), int(corners[(i+1) % 4][1]*img.shape[0]))
        next_id = (i+1)%4
        img = cv.line(img, (round(x[i]), round(y[i])), (round(x[next_id]), round(y[next_id])), (0,0,255), 2)
        # img = cv.circle(img, (int(a),int(b)), 3, colors[i], 2)
    # for i in range(0,len(points),2):
    #     img = cv.circle(img, (round(points[i]*img.shape[1]), round(points[i+1]*img.shape[0])), 4, (0,255,0), 1)
        
    cv.imwrite(output_image_path+".jpg", img)
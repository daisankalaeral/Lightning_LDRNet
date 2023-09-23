import torch
import torchvision
import torch.nn as nn
import albumentations as A
import cv2 as cv
from albumentations.pytorch import ToTensorV2

normal_transform = A.Compose([
    A.Resize(224,224),
    # A.Equalize(p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class Packed_LDRNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.classifier = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(mobilenet.last_channel, 8)
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x,(1,1))
        x = torch.flatten(x,1)

        corners = self.classifier(x)
        return corners

if __name__ == "__main__":
    # Convert .ckpt file into .pth
    # ckpt = torch.load("all/epoch=164-step=49995.ckpt")
    # model_weights = ckpt["state_dict"]
    # for key in list(model_weights):
    #     model_weights[key.replace("backbone_model.", "")] = model_weights.pop(key)
    # del model_weights["border.1.weight"]
    # del model_weights["border.1.bias"]
    # torch.save(model_weights, "ldrnet_weights.pth")
    
    model = Packed_LDRNet()
    # model.load_state_dict(model_weights)
    model.load_state_dict(torch.load("ldrnet_weights.pth"))
    model.eval()
    
    image_path = "test_data/photo_2023-05-14 07.35.53.jpeg"
    img = cv.imread(image_path)
    copied_img = img
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = normal_transform(image = img)["image"]
    
    corners = model(img.unsqueeze(0))[0].detach().numpy()
    print(corners)
    
    x = corners[0::2] * copied_img.shape[1]
    y = corners[1::2] * copied_img.shape[0]
    
    colors = [(0,0,255), (0,255,0), (255,0,0), (255,0,255)]
    for i in range(0,4):
        next_id = (i+1)%4
        copied_img = cv.line(copied_img, (round(x[i]), round(y[i])), (round(x[next_id]), round(y[next_id])), (0,0,255), 2)
        # img = cv.circle(img, (int(a),int(b)), 3, colors[i], 2)
    # for i in range(0,len(points),2):
    #     img = cv.circle(img, (round(points[i]*img.shape[1]), round(points[i+1]*img.shape[0])), 4, (0,255,0), 1)
        
    cv.imwrite("test.jpg", copied_img)
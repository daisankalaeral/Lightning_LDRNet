# Lightning_LDRNet
Implementation of LDRNet with EfficientNet-Lite as backbone in [Pytorch Lightning](https://github.com/Lightning-AI/lightning).

# Dataset Structure
The json file that stores image samples's information needs to be in this structure.
```
train_dataset.json
[
   {
        "image_path": "dataset2/images/VRAwMsIFAy_random_size_0_76_1_color.jpg",
        "corners": [
            [
                42,
                83
            ],
            [
                107,
                86
            ],
            [
                113,
                159
            ],
            [
                46,
                157
            ]
        ],
        "class": "poster"
    },
    {
        "image_path": "midv_2019/DX06_06.tif",
        "corners": [
            [
                97,
                913
            ],
            [
                1917,
                886
            ],
            [
                2039,
                2132
            ],
            [
                152,
                2255
            ]
        ],
        "class": "card"
    }
]
```

# Train
```
python train.py
```

# Configs
Modify configs.py to change parameters.

# Pretrained Weights
Download EfficientNet-Lite's pretrained weights from [here](https://github.com/RangiLyu/EfficientNet-Lite/releases/tag/v1.0).

# Credits
Thanks to [RangiLyu](https://github.com/RangiLyu/EfficientNet-Lite) for providing source code and pretrained weights for EfficientNet-Lite, [niuwagege](https://github.com/niuwagege/LDRNet) for the original implementation of LDRNet.

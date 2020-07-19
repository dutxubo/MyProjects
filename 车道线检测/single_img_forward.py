import torch, os, cv2
from model.model import parsingNet
import scipy.special

import numpy as np
import torchvision.transforms as transforms


from PIL import Image, ImageDraw

'''
git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection
cd Ultra-Fast-Lane-Detection

# 下载culane_18模型
mkdir download 
cd download
wget https://drive.google.com/u/0/uc?export=download&confirm=uI-u&id=1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq

# 将该脚本放置在Ultra-Fast-Lane-Detection root目录下
cd ../
python single_img_forward.py
'''

cls_num_per_lane = 18
griding_num = 200
model_weight = 'download/culane_18.pth'

img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


net = parsingNet(pretrained = False, backbone='18',cls_dim = (griding_num+1, cls_num_per_lane, 4),
                    use_aux=False) # we dont need auxiliary segmentation in testing

state_dict = torch.load(model_weight, map_location='cpu')['model']
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v
net.load_state_dict(compatible_state_dict, strict=False)
net.eval()

def SingleImgFoward(img_path, out_path='./tmp/'):
    img_name = os.path.basename(img_path)
    save_path = os.path.join(out_path, img_name)

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    img = Image.open(img_path)
    img_tensor = img_transforms(img)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        out = net(img_tensor)

    col_sample = np.linspace(0, 800 - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    out_j = out[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == griding_num] = 0
    out_j = loc

    # import pdb; pdb.set_trace()
    vis = cv2.imread(img_path)
    h, w = vis.shape[:2]
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    #ppp = (int(out_j[k, i] * col_sample_w * 1640 / 800) - 1, int(590 - k * 20) - 1)
                    ppp = (int(out_j[k, i] * col_sample_w * w / 800) - 1, int(h - k * h*0.61/18) - 1)
                    cv2.circle(vis,ppp,5,(0,255,0),-1)

    cv2.imwrite( save_path, vis)
    return save_path

if __name__ == '__main__':
    test_img = '/home/songbai.xb/workspace/projects/lane_detection/datasets/test.jpg'
    print(SingleImgFoward(test_img))
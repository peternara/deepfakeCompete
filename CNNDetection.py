import os
import sys
os.chdir("/home/songzhu/PycharmProjects/DeepFake/www.kaggle.com/c/16880/")
sys.path.append("/home/songzhu/PycharmProjects/DeepFake/www.kaggle.com/c/16880/")

import cv2
import torch
import torch.nn
import argparse
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from network.resnet import resnet50

from facenet_pytorch import MTCNN


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_path = "/home/songzhu/PycharmProjects/DeepFake/www.kaggle.com/c/16880/fake_sample_1.png"
# input_path = "/data/songzhu/deepfake/05/faces_2/hknxehfdxt/0.png"
model_path = "/home/songzhu/PycharmProjects/DeepFake/www.kaggle.com/c/16880/peterwang/weights/" +\
             "blur_jpg_prob0.5.pth"
video_path = "/data/songzhu/deepfake/05/dfdc_train_part_5/uuemauhfph.mp4"

face_detector = MTCNN(margin=10, keep_all=True, factor=0.8, device=device, image_size=360).eval()

v_cap = cv2.VideoCapture(video_path)
v_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
v_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

success = v_cap.grab()
success, frame = v_cap.retrieve()


model = resnet50(num_classes=1)
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
model.cuda()
model.eval()

trans = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# raw_img = np.array(Image.open(input_path))
raw_img = Image.fromarray(np.array(Image.open(input_path)), mode="RGB")
# img = trans(raw_img.convert('RGB'))
# frame = cv2.imread(input_path)

# raw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# faces = face_detector(raw_img)
# face = np.transpose( (255*(faces[0].detach().cpu().numpy().squeeze()+1)/2).astype(np.uint8) , [1,2,0])
# img = cv2.resize(face, (360, 360))
img = trans(raw_img)

with torch.no_grad():
    in_tens = img.unsqueeze(0).cuda()
    prob = model(in_tens).sigmoid().item()

print('probability of being synthetic: {:.2f}%'.format(prob * 100))

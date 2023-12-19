# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:41:55 2023

@author: wonchan
"""

import cv2
from ultralytics import YOLO
#from google.colab.patches import cv2_imshow
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils
import ssl
from torchvision import models
import torch.nn.functional as F
import numpy as np
from torch2trt import torch2trt
from torch2trt import TRTModule
import time
import math
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context


def get_image(img_cv):
    img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img).convert("RGB")

# load age estimation model 
def estimation_model( num_classes=101):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False).cuda()
    dim_feats = model.fc.in_features
    model.fc = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model
estimation_model = estimation_model()


# load age estimation model / resnet 50
checkpoint = torch.load('./epoch044_0.pth', map_location='cuda')
# key rewrite
for key in list(checkpoint['state_dict'].keys()):
    if 'last_linear.' in key:
        checkpoint['state_dict'][key.replace('last_linear.', 'fc.')] = checkpoint['state_dict'][key]
        del checkpoint['state_dict'][key]

estimation_model.load_state_dict(checkpoint['state_dict'])
estimation_model.eval().cuda()

x = torch.ones((1, 3, 224, 224))
e_model_trt = torch2trt(estimation_model, [x.cuda()])


# gender classification model
model_gender = models.resnet18(pretrained=False)
num_features = model_gender.fc.in_features
model_gender.fc = nn.Linear(num_features, 2) # binary classification (num_of_class == 2)
model_gender.load_state_dict(torch.load('gender_ResNet18.pth'))
model_gender.eval().cuda()
# convert to tensortrt
model_trt = torch2trt(model_gender, [x.cuda()])

# yolo model
yolo_model = YOLO('yolov.pt')

# get start time
start = time.time()

# load vedio
video_file = './vedio/demo_last.mp4'

# load frame from veio
cap = cv2.VideoCapture(video_file)      
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
gender_cat = ['female', 'male']

# where to save processed vedio 
out = cv2.VideoWriter('output_demo.avi', fourcc, fps, (int(width), int(height)))


# frame count
frame_count=0
cat_list=[]
tmp=0
#

while (cap.isOpened()):          
  ret, frame = cap.read()        
  if ret:                        
    # face detection for yolo
    results = yolo_model(frame)
    
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs

    if boxes.xyxy.shape[0] != 0:
        img_list=[]
        x1_list=[]
        x2_list=[]

        # get x,y coordinate 
        for i in range(boxes.xyxy.shape[0]):
            x1=int(boxes.xyxy[i][0].item()), int(boxes.xyxy[i][1].item())
            x2=int(boxes.xyxy[i][2].item()), int(boxes.xyxy[i][3].item())
           
            image_numpy = cv2.resize(frame[x1[1]:x2[1],x1[0]:x2[0]], (224, 224))
            image_numpy = image_numpy[np.newaxis, :, :, :]
          
        # predict age
            inputs = torch.from_numpy(np.transpose(image_numpy.astype(np.float32), (0, 3, 1, 2)))
            outputs = F.softmax(e_model_trt(inputs.cuda()), dim=-1).cpu().detach().numpy()
            ages = np.arange(0, 101)
            predicted_ages = (outputs * ages).sum(axis=-1)
      
        # predict gender
            gender_outputs = model_trt(inputs.cuda())
            _, preds2 = torch.max(gender_outputs, 1)
            
        # frame count
            if (frame_count%fps==0):
                pass # cat list add
                tmp = [int(frame_count/fps), int(predicted_ages/10)*10, int(preds2[0].item())]
                cat_list.append(tmp)
            
        # 
        # draw age, gender to img
            cv2.rectangle(frame, x1, x2, (255,255,0), 2)
            cv2.putText(frame, f"{int(predicted_ages[0])}. {gender_cat[int(preds2[0].item())]}",x1,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0), 2)
             
    # save frame to videoWriter
    frame_count+=1
    out.write(frame)
  else:
    break

#get end time
end = time.time()
print(f"acceletated running time {end-start:.2f}sec")

cap.release()                          
cv2.destroyAllWindows()

## cutomer classification
time_max=0
cat_list_after = []

# fine time max
for i in range(len(cat_list)):
  if cat_list[i][0] > time_max:
    time_max = cat_list[i][0]
print(time_max)

# time index empty list
for i in range(time_max+1):
  cat_list_after.append([])

# time indexed Category

for i in range(len(cat_list)):
  cat_list_after[cat_list[i][0]].append(cat_list[i][1:])

cat_1_list=[]
cat_2_list=[]
cat_3_list=[]
cat_4_list=[]
cat_5_list=[]
cat_6_list=[]

tmp11=0
tmp22=0
tmp33=0
tmp44=0
tmp55=0
tmp66=0

for i in range(len(cat_list_after)):
  tmp11=0
  tmp22=0
  tmp33=0
  tmp44=0
  tmp55=0
  tmp66=0

  for j in range(len(cat_list_after[i])):
    if (cat_list_after[i][j] == [10, 0] or cat_list_after[i][j] == [20, 0]):
      tmp11+=1
    elif (cat_list_after[i][j] == [10, 1] or cat_list_after[i][j] == [20, 1]):
      tmp22+=1
    elif (cat_list_after[i][j] == [30, 0] or cat_list_after[i][j] == [40, 0]):
      tmp33+=1
    elif (cat_list_after[i][j] == [30, 1] or cat_list_after[i][j] == [40, 1]):
      tmp44+=1
    elif (cat_list_after[i][j] == [50, 0] or cat_list_after[i][j] == [60, 0]):
      tmp55+=1
    elif (cat_list_after[i][j] == [50, 1] or cat_list_after[i][j] == [60, 1]):
      tmp66+=1
    else:
      pass
  # 10_20 woman
  cat_1_list.append(tmp11)
  # 10_20 man
  cat_2_list.append(tmp22)
  # 30_40 woman
  cat_3_list.append(tmp33)
  # 30_40 man
  cat_4_list.append(tmp44)
  # 50_60 woman
  cat_5_list.append(tmp55)
  # 50_60 man
  cat_6_list.append(tmp66)
  
time_axis = []
for i in range(time_max+1):
  time_axis.append(i)


# 각 카테고리 리스트를 딕셔너리로 묶음
categories = {
    '10_20, woman': cat_1_list,
    '10_20, man': cat_2_list,
    '30_40, woman': cat_3_list,
    '30_40, man': cat_4_list,
    '50_60, woman': cat_5_list,
    '50_60, woman': cat_6_list
}

# 선 그래프 그리기
plt.figure(figsize=(10, 2))  # 그래프 사이즈 설정

for category, data in categories.items():
    plt.plot(time_axis, data, label=category)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Customer classification')
plt.legend()
plt.tight_layout()
plt.savefig('cumtomer_classification.png')

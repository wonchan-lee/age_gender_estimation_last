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

ssl._create_default_https_context = ssl._create_unverified_context


def get_image(img_cv):
    img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img).convert("RGB")

# load age estimation model 
def estimation_model( num_classes=101):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
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
estimation_model.eval()

x = torch.ones((1, 3, 224, 224))
e_model_trt = torch2trt(estimation_model, [x])
print(e_model_trt)

#aprint(estimation_model)
#model_trt = torch2trt(estimation_model, [x])
#model_trt = TRTModule()
#model_trt.load_state_dict(torch.load('age_seResNet50.pth'))
#print(model_trt)


# gender classification model
model_gender = models.resnet18(pretrained=True)
num_features = model_gender.fc.in_features
model_gender.fc = nn.Linear(num_features, 2) # binary classification (num_of_class == 2)
model_gender.load_state_dict(torch.load('gender_ResNet18.pth'))
model_gender.eval().cuda()
# convert to tensortrt
model_trt = torch2trt(model_gender, [x.cuda()])
#print(model_trt)

# yolo model
yolo_model = YOLO('yolov.pt')

# load vedio
video_file = './vedio/man3.mp4'

# load frame from veio
cap = cv2.VideoCapture(video_file)      
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
gender_cat = ['female', 'male']

# where to save processed vedio 
out = cv2.VideoWriter('output2.avi', fourcc, fps, (int(width), int(height)))


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
            outputs = F.softmax(estimation_model(inputs), dim=-1).cpu().detach().numpy()
            ages = np.arange(0, 101)
            predicted_ages = (outputs * ages).sum(axis=-1)
      
        # predict gender
            gender_outputs = model_trt(inputs.cuda())
            _, preds2 = torch.max(gender_outputs, 1)
        
        #
        # draw age, gender to img
            cv2.rectangle(frame, x1, x2, (255,255,0), 2)
            cv2.putText(frame, f"{int(predicted_ages[0])}. {gender_cat[int(preds2[0].item())]}",x1,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0), 2)
      
    # save frame to videoWriter
    out.write(frame)
  else:
    break


cap.release()                          
cv2.destroyAllWindows()

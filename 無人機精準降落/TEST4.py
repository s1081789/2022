from __future__ import print_function # 必需置於第一行
# _*_ coding: utf-8 _*_
file_name='PreciseLanding_202212010111'
import torch
device=torch.device('cuda') # 'cuda'/'cpu'，import torch
num_classes=2 # 物件類別數+1(背景)
batch_size=1 # 必為1
variances=[0.1,0.2] # 設定gHat中cx、cy與w、h間的權重(須與訓練的設定值相同)
conf_threshold=0.01 # 將pred_conf大於等於conf_threshold的結果視為候選物件(並命其為scores)，參考值=0.01
top_k=1 # 依scores挑出最大前top_k個後代入NMS，參考值=200
NMS_threshold=0.5 # 將同類別且IoU小於等於NMS_threshold的物件視為不同物件。此值愈小邊界框會愈少，參考值=0.5

# 取得網路
from torch import nn
class SSD(nn.Module):
    def __init__(self):
        super(SSD,self).__init__()
        # block_1：Conv1_1~Conv4_3+ReLU
        self.block_1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1), # [batch_size,64,300,300]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), # [batch_size,64,300,300]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch_size,64,150,150]
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1), # [batch_size,128,150,150]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1), # [batch_size,128,150,150]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch_size,128,75,75]
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1), # [batch_size,256,75,75]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1), # [batch_size,256,75,75]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1), # [batch_size,256,75,75]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True), # [batch_size,256,38,38]
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,38,38]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,38,38] 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,38,38]
            nn.ReLU(inplace=True),
        )
        # Layer learns to scale the L2 normalized features from conv4_3
        self.l2norm=L2Norm(512,20) # 512為輸入的特徵圖個數，20為scale
        # block_2：Pool4~Conv7+ReLU
        self.block_2=nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch_size,512,19,19]
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,19,19]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,19,19]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,19,19]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1), # [batch_size,512,19,19]
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=6,dilation=6), # [batch_size,1024,19,19]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1,stride=1), # [batch_size,1024,19,19]
            nn.ReLU(inplace=True),
        )
        # block_3：Conv8_1~Conv8_2+ReLU
        self.block_3=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=256,kernel_size=1), # [batch_size,256,19,19]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1), # [batch_size,512,10,10]
            nn.ReLU(inplace=True),
        )
        # block_4：Conv9_1~Conv9_2+ReLU
        self.block_4=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=128,kernel_size=1), # [batch_size,128,10,10]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1), # [batch_size,256,5,5]
            nn.ReLU(inplace=True),
        )
        # block_5：Conv10_1~Conv10_2+ReLU
        self.block_5=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1), # [batch_size,128,5,5]
            nn.ReLU(inplace=True),                            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3), # [batch_size,256,3,3]
            nn.ReLU(inplace=True),
        )
        # block_6：Conv11_1~Conv11_2+ReLU
        self.block_6=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1), # [batch_size,128,3,3]                            
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3), # [batch_size,256,1,1]
            nn.ReLU(inplace=True),
        )
        # loc_1
        self.loc_1=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=4*4,kernel_size=3,stride=1,padding=1), # [batch_size,16,38,38]
        )
        # conf_1
        self.conf_1=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=4*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(4*num_classes),38,38]
        )
        # loc_2
        self.loc_2=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=6*4,kernel_size=3,stride=1,padding=1), # [batch_size,24,19,19]
        )
        # conf_2
        self.conf_2=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=6*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(6*num_classes),19,19]
        ) 
        # loc_3
        self.loc_3=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=6*4,kernel_size=3,stride=1,padding=1), # [batch_size,24,10,10]
        )
        # conf_3
        self.conf_3=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=6*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(6*num_classes),10,10]
        ) 
        # loc_4
        self.loc_4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=6*4,kernel_size=3,stride=1,padding=1), # [batch_size,24,5,5]
        )
        # conf_4
        self.conf_4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=6*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(6*num_classes),5,5]
        )       
        # loc_5
        self.loc_5=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*4,kernel_size=3,stride=1,padding=1), # [batch_size,16,3,3]
        )
        # conf_5
        self.conf_5=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(4*num_classes),3,3]
        )   
        # loc_6
        self.loc_6=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*4,kernel_size=3,stride=1,padding=1), # [batch_size,16,1,1]
        )
        # conf_6
        self.conf_6=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(4*num_classes),1,1]
        )   
    def forward(self,x):
        x=self.block_1(x) # [batch_size,512,38,38] (Conv4_3+ReLU輸出)
        n=self.l2norm(x)
        loc1=self.loc_1(n).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,4)
        conf1=self.conf_1(n).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_2(x) # [batch_size,1024,19,19] (Conv7+ReLU輸出)
        loc2=self.loc_2(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,4)
        conf2=self.conf_2(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_3(x) # [batch_size,512,10,10] (Conv8_2+ReLU輸出)
        loc3=self.loc_3(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,4)
        conf3=self.conf_3(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_4(x) # [batch_size,256,5,5] (Conv9_2+ReLU輸出)
        loc4=self.loc_4(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,4)
        conf4=self.conf_4(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_5(x) # [batch_size,256,3,3] (Conv10_2+ReLU輸出)
        loc5=self.loc_5(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,4)
        conf5=self.conf_5(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_6(x) # [batch_size,256,1,1] (Conv11_2+ReLU輸出)
        loc6=self.loc_6(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,4)
        conf6=self.conf_6(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        loc=torch.cat((loc1,loc2,loc3,loc4,loc5,loc6),1) # [batch_size,8732,4]，import torch
        conf=torch.cat((conf1,conf2,conf3,conf4,conf5,conf6),1) # [batch_size,8732,num_classes]，import torch
        return loc,conf
class L2Norm(nn.Module):
    def __init__(self,in_channels,scale):
        super(L2Norm,self).__init__()
        self.in_channels=in_channels
        self.gamma=scale or None
        self.eps=1e-10
        self.weight=nn.Parameter(torch.Tensor(self.in_channels)) # from torch import nn，import torch
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.constant_(self.weight,self.gamma) # from torch import nn 
    def forward(self,x):
        norm=x.pow(2).sum(dim=1,keepdim=True).sqrt()+self.eps
        x=torch.div(x,norm) # import torch
        out=self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)*x
        return out
detector=SSD().to(device)
detector.load_state_dict(torch.load(file_name)) # import torch
detector.eval()

# 建立錨框
feature_scale=[38,19,10,5,3,1] # 預測用的特徵圖尺寸(以像素為單位)
sk=[0.07,0.15,0.33,0.51,0.69,0.87,1.05] # 各預測特徵圖的默認框尺度(相對於輸入影像的比例)，比預測特徵圖的個數多1
aspect_ratio=[[1,2,1/2],[1,2,3,1/2,1/3],[1,2,3,1/2,1/3],[1,2,3,1/2,1/3],[1,2,1/2],[1,2,1/2]] # 各預測特徵圖的縱橫比(須檢查loc、conf的濾波器個數)
abox=[]
import itertools
import math
for i,j in enumerate(feature_scale):
    for m,n in itertools.product(range(j),repeat=2):
        cx=(n+0.5)/j # 等同於cx相對於輸入影像的比例位置(乘以輸入影像尺寸即為cx在輸入影像的像素位置)
        cy=(m+0.5)/j # 等同於cy相對於輸入影像的比例位置(乘以輸入影像尺寸即為cy在輸入影像的像素位置)
        for ar in aspect_ratio[i]:
            abox+=[cx-sk[i]*math.sqrt(ar)/2,cy-sk[i]/math.sqrt(ar)/2,cx+sk[i]*math.sqrt(ar)/2,cy+sk[i]/math.sqrt(ar)/2] # [cxmin cymin cxmax cymax]
        abox+=[cx-math.sqrt(sk[i]*sk[i+1])/2,cy-math.sqrt(sk[i]*sk[i+1])/2,cx+math.sqrt(sk[i]*sk[i+1])/2,cy+math.sqrt(sk[i]*sk[i+1])/2] # [xmin ymin xmax ymax]
anchor=torch.Tensor(abox).view(-1,4).to(device) # [8732,4] (所有錨框的[xmin ymin xmax ymax]，皆相對於輸入影像的比例位置，乘以輸入影像尺寸即為在輸入影像的像素位置)，import torch
anchor.clamp_(max=1, min=0) # 限定最大值為1、最小值0
anchor=anchor*300 # 轉換成輸入影像尺寸

# 設定攝像頭
def gstreamer_pipeline(
    capture_width=3264,
    capture_height=2464,
    display_width=1632,
    display_height=1232,
    framerate=3,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
import cv2 # 匯入cv2套件
import threading
class capCapture:
    def __init__(self,cap):
        self.Frame=[]
        self.status=False
        self.isstop=False
        self.capture=cv2.VideoCapture(cap)
    def start(self):
        #Thread(target=self.queryframe,args=()).start()
        threading.Thread(target=self.queryframe,daemon=True,args=()).start()
    def stop(self):
        self.isstop=True
    def getframe(self):
        return self.Frame.copy()
    def queryframe(self):
        while(not self.isstop):
            self.status,self.Frame=self.capture.read()
        self.capture.release()
csi_camera=capCapture(gstreamer_pipeline(flip_method=0))
csi_camera.start()

# 定義arm_and_takeoff
from pymavlink import mavutil
def arm_and_takeoff(aTargetAltitude):
    print("起飛前檢查. . .")
    while not vehicle.is_armable: # 檢查飛控是否啟動、有無GPS fix、卡曼濾波器
        print("等待飛機初始化")
        time.sleep(1)
    print("切換至GUIDED模式. . .") 
    vehicle.mode=VehicleMode("GUIDED")
    vehicle.armed=True # 解鎖
    while not vehicle.armed:
        print("等待解鎖. . .")
        time.sleep(1)
    print("起飛!!!. . .")
    vehicle.simple_takeoff(aTargetAltitude)
    while True:
        print("相對於home點的高度:",vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95:
            print("到達目標高度. . .")
            break
        time.sleep(1)

# 定義send_ned_velocity
def send_ned_velocity(velocity_x,velocity_y,velocity_z,duration):
    msg=vehicle.message_factory.set_position_target_local_ned_encode(
            0,
            0, 0,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,
            0, 0, 0,
            velocity_x,velocity_y,velocity_z,
            0, 0, 0,
            0, 0)
    #循環發送幾次
    for x in range(0,duration):
        #發送指令
        vehicle.send_mavlink(msg)
        time.sleep(1)

from dronekit import connect,VehicleMode,LocationGlobalRelative

# 連接無人機
vehicle=connect('/dev/ttyTHS1',wait_ready=True,baud=57600)

# 起飛
import time
arm_and_takeoff(7)

# 飛至目的地上方
point1=LocationGlobalRelative(24.2259901,120.5784626,7) # 北緯20.203541，東經120.564910，相對home點高20公尺
vehicle.simple_goto(point1,groundspeed=3) # 空速default值=3m/s。simple_goto只發送指令，不判斷有沒有到達目標航點，可被其他後續指令打斷
time.sleep(10)

# 執行精準降落
from torchvision import transforms
transforms=transforms.Compose([transforms.Resize((300,300)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) # ToTensor將影像像素歸一化至0~1(直接除以255)，from torchvision import transforms
from PIL import Image
import numpy


while True:
    if vehicle.location.global_relative_frame.alt<=1:
        print("低於1公尺 進LAND")
        break
    img_cv=csi_camera.getframe() # 讀取影像，img：[300,300,3]
    x0,y0,x2,y2=0,0,img_cv.shape[1],img_cv.shape[0] # 避免一開始就沒有偵測到物件導致計算delta_x與delta_y時沒有x0、y0、x2、y2
    I=Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB)) # from PIL import Image，opencv轉PIL.Image
    img=transforms(I) # [3,300,300]，from torchvision import transforms
    img=img.unsqueeze(0) # [1,3,300,300]
    img=img.to(device)

    # 預測結果
    pred_loc,pred_conf=detector(img) # pred_loc：[batch_size,8732,4]，pred_conf：[batch_size,8732,num_classes]
    pred_conf=pred_conf.view(batch_size,pred_conf.shape[1],num_classes).transpose(2,1) # [batch_size,num_classes,8732]
    for i in range(batch_size):
        pred_cx=(anchor[:,0]+anchor[:,2])/2+pred_loc[i][:,0]*((anchor[:,2]-anchor[:,0])*variances[0]) # 預測的cx，[8732]
        pred_cy=(anchor[:,1]+anchor[:,3])/2+pred_loc[i][:,1]*((anchor[:,3]-anchor[:,1])*variances[0]) # 預測的cy，[8732]
        pred_w=torch.exp(pred_loc[i][:,2]*variances[1])*(anchor[:,2]-anchor[:,0])  # 預測的w，[8732]，import torch
        pred_h=torch.exp(pred_loc[i][:,3]*variances[1])*(anchor[:,3]-anchor[:,1])  # 預測的h，[8732]，import torch
        pred_bbox=torch.stack((pred_cx-pred_w/2,pred_cy-pred_h/2,pred_cx+pred_w/2,pred_cy+pred_h/2),dim=1) # 預測的邊界框，[8732,4]，[xmin ymin xmax ymax]，import torch
        for j in range(1,num_classes): # 1~(num_classes-1)
            c_mask=pred_conf[i][j].ge(conf_threshold) # 每個錨框對第j個物件的預測置信值是否大於等於conf_threshold，True/False，[8732]
            scores=pred_conf[i][j][c_mask] # 針對第i個batch的第j個物件，挑出大於conf_threshold的預測置信值，並命其為scores，[如533]
            if scores.size(0)==0:
                continue
            l_mask=c_mask.unsqueeze(1).expand_as(pred_bbox)
            boxes=pred_bbox[l_mask].view(-1,4) # 針對第i個batch的第j個物件，找出scores(即大於conf_threshold的置信值)對應的預測邊界框，[如533,4]
            area=torch.mul(boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1]) # import torch，針對第i個batch的第j個物件，計算出scores(即大於conf_threshold的置信值)所對應的預測邊界框的面積，[如533]

            # NMS
            _,idx=scores.sort(0) # idx：scores由小而大排列並取得位置編號，[如533]
            idx=idx[-top_k:] # 挑出scores最大前top_k個的位置編號，[top_k]
            best_idx=[] # 儲存某類別(不含背景)經NMS後的位置編號
            while idx.numel()>0:
                b=idx[-1] # 目前scores中最大值的位置編號
                x0=int(boxes[b,0]/300*img_cv.shape[1])
                y0=int(boxes[b,1]/300*img_cv.shape[0])
                x2=int(boxes[b,2]/300*img_cv.shape[1])
                y2=int(boxes[b,3]/300*img_cv.shape[0])
                cv2.rectangle(img_cv,(x0,y0),(x2,y2),(255,0,0),3)
                best_idx.append(b.item())
                if idx.size(0)==1:
                    break
                idx=idx[:-1] # 移除idx中最後一個位置編號(即目前scores中值最大的位置編號)，idx內的位置編號數減少1個
                min_x=torch.clamp(torch.index_select(boxes[:,0],0,idx),min=boxes[b,0].item()) # 交集部分的最小x，[top_k]、[top_k-1]、...[1]，import torch
                min_y=torch.clamp(torch.index_select(boxes[:,1],0,idx),min=boxes[b,1].item()) # 交集部分的最小y，[top_k]、[top_k-1]、...[1]，import torch
                max_x=torch.clamp(torch.index_select(boxes[:,2],0,idx),max=boxes[b,2].item()) # 交集部分的最大x，[top_k]、[top_k-1]、...[1]，import torch
                max_y=torch.clamp(torch.index_select(boxes[:,3],0,idx),max=boxes[b,3].item()) # 交集部分的最大y，[top_k]、[top_k-1]、...[1]，import torch
                area_inter=torch.clamp(max_x-min_x,min=0.0)*torch.clamp(max_y-min_y,min=0.0) # import torch
                IoU=area_inter/(torch.index_select(area,0,idx)+area[b]-area_inter) # scores中最大值的邊界框與剩餘邊界框的IoU，import torch
                idx=idx[IoU.le(NMS_threshold)] # 在剩餘邊界框中保留IoU小於等於NMS_threshold的邊界框
   # cv2.imshow('Frame',img_cv)
   # k=cv2.waitKey(1) 

    # 計算x與phi
    delta_x=-(x0+x2)/2+img_cv.shape[1]/2 # 左右像素差，正表示H在右方
    delta_y=-(y0+y2)/2+img_cv.shape[0]/2 # 前後像素差，正表示H在後方

    #移動
    vx=numpy.sign(delta_x)*numpy.clip(abs(1/(x2-x0)*delta_x)*1,0,0.25)
    vy=numpy.sign(delta_y)*numpy.clip(abs(1/(y2-y0)*delta_y)*1,0,0.25)
    vz=0.2
    send_ned_velocity(vx,vy,vz,1)
    print(vx,vy,vz)

vehicle.mode=VehicleMode("LAND")   #著地模式 切換成land模式
print("LAND")

vehicle.close()
csi_camera.stop()

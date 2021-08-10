import cv2
import torch
from model import PositionFinder
import helper
import numpy as np
import time
from torchvision import transforms
vid = cv2.VideoCapture(0)

img_width = 256
device = 'cuda'
model = PositionFinder(img_width)
model.load_state_dict(torch.load('model.m'))
model.to(device)
details = torch.Tensor([[0.9,1]])
bboxes = torch.Tensor([0.0, 0.0, 1.0, 1.0])
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
font = cv2.FONT_HERSHEY_SIMPLEX

minval, maxval = 0, 512

start_time = time.time()
fps_update = 1 # displays the frame rate every 1 second
counter = 0
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    print(frame.shape)

    if ret == True:
        crop_from = (frame.shape[1] - frame.shape[0]) // 2
        # img = frame[:, crop_from:crop_from+frame.shape[0]]
        # img = cv2.resize(frame, (img_width, img_width), cv2.INTER_CUBIC)

        #normalization
        # img = img / 255
        # img = (img - mean) / std
        # print(img.shape)
        # img = img.swapaxes(1,2).swapaxes(0,1)
        # print(img.shape)
        tran = transforms.ToTensor()
        resi = transforms.Resize((img_width,img_width))
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = tran(frame)
        img = resi(img)
        img = norm(img)
        # img = resi(frame)
        # img = norm(frame)
        images = torch.Tensor([np.array(img)])
        positions = []
        positions_expected = []
        
        with torch.no_grad():
            images, details, bboxes = images.to(device), details.to(device), bboxes.to(device)
            
            logps = model.forward(images, details, bboxes)
            logps_denormalized = logps 
            for body_index in range(5):
                xyz = []
                xyz_e = []
                for pos_index in range(2):
                    pos = logps_denormalized[0][body_index][pos_index]
                    pos = pos * (maxval - minval) + minval

                    xyz.append(pos)
                positions.append(xyz)
        

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice

        for i, pos in enumerate(positions):
            # x = int(pos[0].item() // (512/frame.shape[0]))
            # y = int(pos[1].item() // (512/frame.shape[1]))
            x = int(pos[0].item())
            y = int(pos[1].item())
            print(x,y, pos)
            frame = cv2.circle(frame, (x,y), radius=10, color=(0, 0, 255), thickness=-1)
            frame = cv2.putText(frame, helper.get_label(i), (x,y), color=(0, 0, 255), fontFace=font, fontScale=1, thickness=2)

        counter+=1
        if (time.time() - start_time) > fps_update :
            fps = counter / (time.time() - start_time)
            print(fps)
            cv2.setWindowTitle('frame', "FPS " + str(fps))
            counter = 0
            start_time = time.time()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


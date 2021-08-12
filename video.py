import cv2
import torch
from model import PositionFinder, BoundingBoxFinder
import model
import helper
import numpy as np
import time
from torchvision import transforms
vid = cv2.VideoCapture(0)

img_width = 256
device = 'cuda'
pmodel = PositionFinder(img_width)
bmodel = BoundingBoxFinder(img_width)
pmodel.load_state_dict(torch.load('model.m'))
bmodel.load_state_dict(torch.load('bmodel.m'))
pmodel.to(device)
bmodel.to(device)
details = torch.Tensor([[0.9,1]]).to(device)
bboxes = torch.Tensor([0.0, 0.0, 1.0, 1.0])
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
font = cv2.FONT_HERSHEY_SIMPLEX

minval, maxval = 0, 512

start_time = time.time()
fps_update = 1 # displays the frame rate every 1 second
counter = 0
tran = transforms.ToTensor()
resi = transforms.Resize((img_width,img_width))
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    if ret == True:
        img = tran(frame)
        img = resi(img)
        img = norm(img)
        images = torch.Tensor([np.array(img)])
        positions = []
        positions_expected = []
        
        with torch.no_grad():
            images = images.to(device)
            
            bboxes = bmodel.forward(images)
            x = int(max(0, min(1, bboxes[0][0])) * 512)
            y = int(max(0, min(1, bboxes[0][1])) * 512)
            w = int(max(0, min(1, bboxes[0][2])) * 512 - x) 
            h = int(max(0, min(1, bboxes[0][3])) * 512 - y) 
            crop_img = frame[y:y+h, x:x+w]
            img2 = tran(crop_img)
            img2 = resi(img2)
            img2 = norm(img2)
            images2 = torch.Tensor([np.array(img2)])
            images2 = images2.to(device)

            logps = pmodel.forward(images2, details, bboxes)
            logps_denormalized = model.align_result_out_of_bounding_boxes(logps[0], bboxes[0])
            for body_index in range(5):
                xyz = []
                xyz_e = []
                for pos_index in range(2):
                    pos = logps_denormalized[body_index][pos_index]
                    pos = pos * (maxval - minval) + minval

                    xyz.append(pos)
                positions.append(xyz)
        

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice

        for i, bbox in enumerate(bboxes):
            x1 = int(bbox[0].item() * 512) 
            y1 = int(bbox[1].item() * 512)
            x2 = int(bbox[2].item() * 512)
            y2 = int(bbox[3].item() * 512)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

        for i, pos in enumerate(positions):
            # x = int(pos[0].item() // (512/frame.shape[0]))
            # y = int(pos[1].item() // (512/frame.shape[1]))
            x = int(pos[0].item())
            y = int(pos[1].item())
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


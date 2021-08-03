import cv2
import torch
from model import Network
import numpy as np

vid = cv2.VideoCapture(0)

img_width = 256
model = Network(img_width)
model.load_state_dict(torch.load('model.m'))
details = torch.Tensor([[0.9,1]])
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

minval, maxval = -158.6369, 545.9897

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    frame = cv2.resize(frame, (img_width, img_width), cv2.INTER_CUBIC)

    #normalization
    img = frame / 255
    img = (img - mean) / std
    images = torch.tensor(np.array([img.T]).astype('float32'))
    positions = []
    positions_expected = []
    
    with torch.no_grad():
        logps, estimations = model.forward(images, details)
        logps_denormalized = logps 
        for body_index in range(22):
            xyz = []
            xyz_e = []
            for pos_index in range(2):
                pos = logps_denormalized[0][body_index*2+pos_index]
                pos = pos * (maxval - minval) + minval

                xyz.append(pos)
            positions.append(xyz)
      

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    for pos in positions:
        x = int(pos[0].item()) // 2
        y = img_width - int(pos[1].item()) // 2
        frame = cv2.circle(frame, (x,y), radius=2, color=(0, 0, 255), thickness=-1)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
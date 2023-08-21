
import torch
import numpy as np
import cv2
from centroidtracker import CentroidTracker


tracker = CentroidTracker(maxDisappeared=10, maxDistance=10) #(maxDisappeared=80, maxDistance=90) maxDisappeared=10 is good for us

# device='cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = torch.hub.load('ultralytics/yolov5', 'custom', path="Weapon_14Aug.pt")#, force_reload=True)
model5 = torch.hub.load('WongKinYiu/yolov7', 'custom', "weapon_seperated_13jul.pt")#weapon_seperated_13jul#weapon_3_13Dec
# model = load_model(model_name)
# model5.to(device)
classes = model5.names


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]  # Extract the x1 coordinates from the boxes
        y1 = boxes[:, 1]  # Extract the y1 coordinates from the boxes
        x2 = boxes[:, 2]  # Extract the x2 coordinates from the boxes
        y2 = boxes[:, 3]  # Extract the y2 coordinates from the boxes



        area = (x2 - x1 + 1) * (y2 - y1 + 1)# Compute the area of each box
        idxs = np.argsort(y2)# Sort the boxes based on the y2 coordinates

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i) # Add the index of the current box to the list of picked boxes
            # Calculate the intersection coordinates of the current box with the previous boxes
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)# Compute the width of the intersection area
            h = np.maximum(0, yy2 - yy1 + 1)# Compute the height of the intersection area

            overlap = (w * h) / area[idxs[:last]]# Calculate the overlap ratio between the intersection area and the area of previous boxes

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))# Remove the indexes of boxes that have overlap greater than the threshold

        return boxes[pick].astype("int")# Return the picked boxes as integer coordinates
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

def score_frame(frame):
        
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """

        model5.to(device)
        frame = [frame]
        # frame=frame.to(device)
        results = model5(frame)
        # print('........outside.............')
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        # torch.cuda.free(model5)
        torch.cuda.empty_cache()
        return labels, cord
def plot_boxes(results, frame):
        
        # print('wwwwwwwwwwwwwwwwwwwwwwwwwwwww')
        c=''
        text=''
        bbox_index=''
        

        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        cord = cord.cpu().detach().numpy()###############3
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
     
        rects = []
        for i in range(n):
            row = cord[i]
            print('....before weapon..',row[4])
           
            if row[4] >= 0.65: #0.75 #65 with 2 counter (no wrong prediction for weapon3_13dec)
               
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
              
                
                if classes[int(labels[i])] == "Mobile":
                    continue
                ################ skip detection more than 25% of img ###############
                bboxarea=(x2 - x1) * (y2-y1)
                totalarea=x_shape*y_shape
                percent = int((bboxarea/totalarea)*100)
              
                if percent > 25:
                    continue
                ################ skip detection more than 25% of img ###############
                idx = int(labels[i])
                c=classes[int(labels[i])]
                bbox_index=i
                (H, W) = frame.shape[:2]
                det_box = cord[i, 0:4] * np.array([W, H, W, H])
                # det_box=det_box.numpy()
                imageCopy = frame.copy()
                (startX, startY, endX, endY) = det_box.astype("int")
                rects.append(det_box)
                
                
            # detection_index.append(i)
        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)#0.3


        objects = tracker.update(rects)

        
        for (number,(objectId, bbox)) in enumerate(objects.items()):
                x1, y1, x2, y2 = bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

        #################################
                cx =int((x1+x2)/2.0)
                cy = int((y1+y2)/2.0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                text = "ID: {}".format(objectId)
                # bbox_index=detection_index[number]
                cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                # print('rects',rects,'track content',objects.items())
                # print('cord',cord,'labels',labels,"track",(objectId, bbox),'index',number)
                # print('length',len(rects),'length of det',len(detection_index))

        return text,c,bbox_index


        
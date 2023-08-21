
import torch
import numpy as np
import cv2
from centroidtracker import CentroidTracker
tracker = CentroidTracker(maxDisappeared=20, maxDistance=50) #(maxDisappeared=80, maxDistance=90)
device='cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt")#, force_reload=True)
model3 = torch.hub.load('WongKinYiu/yolov7', 'custom', "dog_poop_30Jan.pt")
# model3.to(device)
classes = model3.names

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))
def score_frame(frame):

        model3.to(device)
        frame = [frame]
        # frame=frame.to(device)
        results = model3(frame)
        # print('........outside.............')
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        # torch.cuda.free(model3)
        return labels, cord
def plot_boxes(results, frame):
        text = ''
        c=''
        co2=''
        bbox_index=''
        labels, cord = results
        # cord = cord.cpu().detach().numpy()###############3
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        rects = []
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.80:  # 0.3
                # a = float(row[4])
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                if classes[int(labels[i])] == "No-Poop":
                        continue
                ################ skip detection more than 25% of img ###############
                bboxarea=(x2 - x1) * (y2-y1)
                totalarea=x_shape*y_shape
                percent = int((bboxarea/totalarea)*100)
                # print('percent...',percent)
                if percent > 75:
                    continue
                ################ skip detection more than 25% of img ###############
                # bgr = (0, 255, 0)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                # c= classes[int(labels[i])]
                # cv2.putText(frame, c+str("%.2f" % a), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2) #0.9
                # # print('...................',c)
                idx = int(labels[i])
                print('....idx..', classes[int(labels[i])])
                co2=classes[int(labels[i])]
                bbox_index=i
                # if classes[int(labels[i])] == "Person":
                #         continue
                # print('cord',cord)
                (H, W) = frame.shape[:2]
                # print('/',cord[i, 0:4])
                det_box = cord[i, 0:4] * np.array([W, H, W, H])
                det_box = det_box.numpy()
            
                imageCopy = frame.copy()
                (startX, startY, endX, endY) = det_box.astype("int")
                # cv2.rectangle(imageCopy, (startX, startY), (endX, endY), (0, 0, 0), 2)
                # cv2.imshow("rough", imageCopy)
                rects.append(det_box)
        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        # print('before rects',rects)
        rects = non_max_suppression_fast(boundingboxes, 0.3)  # 0.3
        # print('after rects',rects)
        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            #################################
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            # cv2.imwrite('images/poop_.jpg', frame)

        return text,co2,bbox_index
        # x_shape, y_shape = frame.shape[1], frame.shape[0]
        # for i in range(n):
        #     row = cord[i]
        #     if row[4] >= 0.98: #0.3
        #         a = float(row[4])
        #         x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        #         bgr = (0, 255, 0)
        #         # cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        #         c= classes[int(labels[i])]
        #         # print('lmop...............',c)
        #         # cv2.putText(frame, c+str("%.2f" % a), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2) #0.9
        #         # print('...................',c)
        #
        #
        #
        # return c


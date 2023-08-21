
import torch
import numpy as np
import cv2
from centroidtracker import CentroidTracker
tracker = CentroidTracker(maxDisappeared=10, maxDistance=10) #(maxDisappeared=80, maxDistance=90)
# device='cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt")#, force_reload=True)
model6 = torch.hub.load('WongKinYiu/yolov7', 'custom', "window_2class_25April.pt")

classes = model6.names
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
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model6.to(device)
        frame = [frame]
        # frame=frame.to(device)
        results = model6(frame)
        # print('........outside.............')
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        # torch.cuda.free(model6)
        return labels, cord


def plot_boxes(results, frame):
        text = ''
        co8=''
        bbox_index=''
        labels, cord = results
        cord = cord.cpu().detach().numpy()######## gpu #######3
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        # rects = []
        # x_shape, y_shape = frame.shape[1], frame.shape[0]
        # print('x_shape',x_shape)
        y=''
        # print('beforeeee',y)
        for i in range(n):
            row = cord[i]
            # print('row',row)
            
            # print('.........eaves..........',classes[int(labels[i])])
            if row[4] >= 0.50: #0.3
                # print('.........eaves..........',classes[int(labels[i])])
                a = float(row[4])
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                ################ skip detection more than 25% of img ###############
                # bboxarea=(x2 - x1) * (y2-y1)
                # totalarea=x_shape*y_shape
                # percent = int((bboxarea/totalarea)*100)
                # print('percent',percent)
                # if percent > 25:
                #     continue
                ################ skip detection more than 25% of img ###############
                # print('x1',x1)
                # if class_to_label(labels[i]) in ['front','left','right']:
                #     bgr = (0, 0, 0)
                # if class_to_label(labels[i])=='Baby':
                    # bgr = (240, 174, 70)
                # if classes(labels[i])!='Person':
                #     continue
                # if classes(labels[i])=='Cat':
                #     bgr = (120, 59, 103)
                # if classes(labels[i])=='Dog':
                #     bgr = (150, 40, 13)
                # bgr = (0, 0, 0)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
                # cv2.putText(frame, classes[int(labels[i])]+str("%.2f" % a), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2) #0.9
                co8=classes[int(labels[i])]
                bbox_index=i
                # print(',,,,,,,,,,,,,,',labels[i])
                # print('...................',classes[int(labels[i])])
        # print('here is the final list',y)
        return co8,bbox_index
        # x_shape, y_shape = frame.shape[1], frame.shape[0]
        # for i in range(n):
        #     row = cord[i]
        #     if row[4] >= 0.60: #0.3
        #
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
        # return c#frame


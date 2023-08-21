
import torch
import numpy as np
import cv2
from centroidtracker import CentroidTracker
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device='cpu'
# model = torch.hub.load('ultralytics/yolov5', 'custom', path="baby.pt")#, force_reload=True)
model2 = torch.hub.load('WongKinYiu/yolov7', 'custom', "baby_pet_28Dec.pt")
# model2.to(device)
classes = model2.names
tracker = CentroidTracker(maxDisappeared=10, maxDistance=10) #(maxDisappeared=80, maxDistance=90)
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

        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        model2.to(device)
        frame = [frame]
        # frame=frame.to(device)
        results = model2(frame)
        # print('........outside.............')
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        # torch.cuda.free(model2)
        torch.cuda.empty_cache()
        return labels, cord


def plot_boxes(results, frame):
        # print('inside Baby_pet file')
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        c = ''
        d = 'Normal'
        text = ''
        bbox_index=''
        run_count = 0
        labels, cord = results
        cord = cord.cpu().detach().numpy()###############3
        # print('labels',results)
        # print('cord',cord)
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        predictions = []
        predictions_index=[]
        rects = []
        # for i in range(n):
        #     row = cord[i]
        #     if row[4] >= 0.50: #0.3
        #         a = float(row[4])
        #         x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        #         bgr = (0, 255, 0)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

        # rects = []
        # print('....cat dog..', classes[int(labels[i])])
        for i in np.arange(0, len(labels)):
                # print('cord', cord)
                confidence = cord[i, 4]  # detections[0, 0, i, 2]
                # print('....idx..', classes[int(labels[i])])
                # print('....cat dog..', classes[int(labels[i])],' confidence ', confidence)
                if confidence > 0.60:#70
                        idx = int(labels[i])
                        c = classes[int(labels[i])]
                        predictions.append(c)
                        # x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                        # print('....idx..', classes[int(labels[i])])
                        # print('....idx..',classes[int(labels[i])])
                        if (classes[int(labels[i])] == "Baby") and (confidence < 0.70):
                              continue
                        if (classes[int(labels[i])] == "Cat") and (confidence < 0.78):#75
                            continue
                        if (classes[int(labels[i])] == "Dog") and (confidence < 0.78):#75
                            continue
                        
                        # print('*************detection inside the loop..', classes[int(labels[i])], confidence)
                        # print('cord',cord)
                        (H, W) = frame.shape[:2]
                        # print('/',cord[i, 0:4])
                        det_box = cord[i, 0:4] * np.array([W, H, W, H])
                        # det_box = det_box.numpy()
                 
                        imageCopy = frame.copy()
                        (startX, startY, endX, endY) = det_box.astype("int")
                        # print('startx',startX)
                        # print('endx',endX)
                        # print('starty',startY)
                        # print('endy',endY)
                        ################ skip detection more than 25% of img ###############
                        bboxarea2=(endX - startX) * (endY-startY)
                        totalarea=x_shape*y_shape
                        # print('bboxarea2',bboxarea2)
                        # print('total area',totalarea)
                        percent = int((bboxarea2/totalarea)*100)
                        # print('percent',percent)
                        if percent > 75:#85
                            continue
                        ################ skip detection more than 25% of img ###############
                        # print('startX',startX)
                        # cv2.rectangle(imageCopy, (startX, startY), (endX, endY), (0, 0, 0), 2)
                        # cv2.imshow("rough", imageCopy)
                        predictions_index.append(i)
                        rects.append(det_box)
        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        # print('before rects',rects)
        rects = non_max_suppression_fast(boundingboxes, 0.3)
        # print('after rects',rects)
        objects = tracker.update(rects)
        # temp=[]
        # print('need to check ',len(objects.items()))
        for (objectId, bbox) in objects.items():
                x1, y1, x2, y2 = bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                #################################
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)

                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                text = "ID: {}".format(objectId)
                # cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                # cv2.imwrite('images/baby'+str(length)+'.jpg',frame)
                # print("text", text)
                # temp.append(text)
        # print('predictions before', predictions)
        if (len(predictions) > 0) and ("Person" not in predictions):

                # print('predictions', predictions)
                return text,predictions,predictions_index  # ,len(objects.items())
        else:
                text = ''
                return text,predictions,predictions_index
# def plot_boxes(results, frame):
#         # print('inside Baby_pet file')
#         """
#         Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
#         :param results: contains labels and coordinates predicted by model on the given frame.
#         :param frame: Frame which has been scored.
#         :return: Frame with bounding boxes and labels ploted on it.
#         """
#         c=''
#         d='Normal'
#         run_count=0
#         labels, cord = results
#         # print('labels',results)
#         # print('cord',cord)
#         n = len(labels)
#         # print('n',n)
#         x_shape, y_shape = frame.shape[1], frame.shape[0]
#         predictions=[]
#         for i in range(n):
#             # print('i',i)
#             row = cord[i]
#             # print('////////////////', int(labels[i]))
#             if row[4] >= 0.80: #0.3
#                 a = float(row[4])
#                 x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
#                 bgr = (0, 255, 0)
#                 # cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
#                 c= classes[int(labels[i])]
#                 # print('////////////////',int(labels[i]))
#                 # if (((int(labels[i]))==0) or ((int(labels[i]))==2))  and (row[4]>=0.80):
#                 if ((int(labels[i])) == 0)  and (row[4] >= 0.80):
#                         predictions.append(c)
#                 if((int(labels[i])) == 1):
#                         predictions.append(c)
#                 # print(c)
#                 # cv2.putText(frame, c+str("%.2f" % a), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2) #0.9
#                 # print('...................',c)
#
#         print(predictions)
#         if (len(predictions)>0) & ("Person" not in predictions):
#                 d='Run'
#         return d#frame

""" frame = cv2.imread("D:/MOJO/weapon/extras/pistol.png")
results = score_frame(frame)
frame = plot_boxes(results, frame)
cv2.imwrite('detection_results.jpg',frame) """
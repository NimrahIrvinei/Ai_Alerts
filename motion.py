import numpy as np
import cv2


def motion_detect(frame2,previous_frame):
    r = True
    text = 'SAFE'
    color = (0, 255, 0)
    cv_font = cv2.FONT_HERSHEY_SIMPLEX
    # frame_count += 1
    img_brg = np.array(frame2)
    img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)
    # 2. Prepare image; grayscale and blur
    prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5,5), sigmaX=0)
    # 3. Set previous frame and continue if there is None
    # if (previous_frame == 'img'):
    #     # print('inside')
    #     previous_frame = prepared_frame
    #     r = False
    # if r == True:

    
    diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
   
    # print('outside')
    # cv2.imwrite('rough/diff_frame_'+str(j)+'.jpg',diff_frame)
    # j+=1
    previous_frame = prepared_frame
    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)######################### last changing
    # 5. Only take different areas that are different enough (>20 / 255)
    thresh_frame = cv2.threshold(src=diff_frame, thresh=100, maxval=255, type=cv2.THRESH_BINARY)[1] #20
    # 6. Find The Contours Boxes From Frame
    contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # print('......frame_count',frame_count)
    # print('.......danger_count',danger_frame)
    # 7. Segment Changes
    if len(contours) != 0:
        # print('///////////////',len(contours))
        # danger_frame = frame_count
        c = max(contours, key = cv2.contourArea)
        # print('***contourArea',cv2.contourArea(c))
        if cv2.contourArea(c) > 200: #100 #500 is good
            (x, y, w, h) = cv2.boundingRect(c)
            # cv2.rectangle(img=img_brg, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
        # get boundary of this text
            text = 'DANGER'
            color = (0, 0, 255)
        # print('frame_count',frame_count)
        # print('danger_count',danger_frame)

        # print(',,,,,,,,,,,,,,Danger,,,,,,,,,,,,,,,')
    # elif frame_count - danger_frame > 30:
    #     # print('.......safe......')
    #     text = 'SAFE'
    #     color = (0, 255, 0)



    # textsize = cv2.getTextSize(text, cv_font, 1, 2)[0]
    # textX = round((640 - textsize[0]) / 2)
    # textY = round(textsize[1] + 10)



    #print('***************',text)
    # print('-------inside', previous_frame == prepared_frame)
    # int+=1
    return text,previous_frame
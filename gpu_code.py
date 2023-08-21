from ast import Break
import pickle
from email import message
# from aiohttp import web
# import socketio
from PIL import Image
import traceback
import base64
import os, os.path
import io
import PIL.Image as Image
from array import array
import cv2
from io import BytesIO
import numpy as np
import requests
# import torch 

# import gesture
# from videoffmepg import create_video,backup_clip
# import weapon_detect
import weapon_tracking
import baby_pet
import dog_poop
import parcel
import fire
import window
import motion
import veh_and_ani
import jumping
import time
# import mediapipe as mp
import face_recognition
from collections import Counter
from threading import Thread
# import asyncio
import base64
import json
import random
from datetime import datetime
import re
# from flask import Flask
from werkzeug.routing import BaseConverter

from flask import Flask, render_template, flash, redirect, url_for,request,Response
import xml.etree.ElementTree as ET
# bbox_index=None



def resize_and_pad_image(frame, target_size):
    # Get the dimensions of the frame
    frame_height, frame_width = frame.shape[:2]
    print('width',frame_width,'....height',frame_height)

    # Calculate the target size and aspect ratio
    target_width, target_height = target_size
    aspect_ratio = frame_width / frame_height

    # Determine the resize dimensions based on aspect ratio
    if aspect_ratio > 1:
        # Landscape orientation
        new_width = target_width
        new_height = round(target_width / aspect_ratio)
    else:
        # Portrait or square orientation
        new_height = target_height
        new_width = round(target_height * aspect_ratio)

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a blank image of the target size
    padded_frame = np.full((target_height, target_width, 3), 255, dtype=np.uint8)

    # Calculate the position to paste the resized frame
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2

    # Paste the resized frame onto the padded frame
    padded_frame[y:y + new_height, x:x + new_width] = resized_frame

    return padded_frame


def create_xml_annotation(filename, class_name,width,height,depth, xmin, ymin, xmax, ymax):
    # print('************!!!!!!!!**************')
    root = ET.Element("annotation")
    
    folder = ET.SubElement(root, "folder")
    folder.text = "annotations"
    
    filename_element = ET.SubElement(root, "filename")
    text_path1=filename+'.jpg'
    filename_element.text = text_path1
    
    path_element = ET.SubElement(root, "path")
    text_path2='/root/livestream/dataset/'+filename+'.jpg'
    path_element.text = text_path2  # Leave the path empty if the file is not present in the directory
    
    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    
    size = ET.SubElement(root, "size")
    width_element = ET.SubElement(size, "width")
    width_element.text = str(width)
    height_element = ET.SubElement(size, "height")
    height_element.text = str(height)
    depth_element = ET.SubElement(size, "depth")
    depth_element.text = str(depth)
    
    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"
    
    object_element = ET.SubElement(root, "object")
    
    name = ET.SubElement(object_element, "name")
    name.text = class_name
    
    pose = ET.SubElement(object_element, "pose")
    pose.text = "Unspecified"
    
    truncated = ET.SubElement(object_element, "truncated")
    truncated.text = "0"
    
    difficult = ET.SubElement(object_element, "difficult")
    difficult.text = "0"
    
    bndbox = ET.SubElement(object_element, "bndbox")
    
    xmin_element = ET.SubElement(bndbox, "xmin")
    xmin_element.text = str(xmin)
    
    ymin_element = ET.SubElement(bndbox, "ymin")
    ymin_element.text = str(ymin)
    
    xmax_element = ET.SubElement(bndbox, "xmax")
    xmax_element.text = str(xmax)
    
    ymax_element = ET.SubElement(bndbox, "ymax")
    ymax_element.text = str(ymax)
    
    tree = ET.ElementTree(root)
    try:
        tree.write('annotations/'+filename + ".xml")
    except Exception as e:
            file_error = open("error_anno2.txt", "w")
            file_error.write(e)
            file_error.close()



def save_img(labels,cord,name,class_name,test_frame,frame2_,bbox_index):
    # class_list=['Poop','Pooping','No-Poop','Baby','Person','Cat','Dog','Pistol','Rifle','Mobile','Fire','Smoke','Stove','window','door','Parcel','Not_Parcel']
    n = len(labels)
    x_shape, y_shape, depth= test_frame.shape[1], test_frame.shape[0],test_frame.shape[2]
    # for i in range(n):
    if bbox_index !='':
        row = cord[bbox_index]
        # if row[4] >= 0.75:
        a = float(row[4])
        s1, t1, s2, t2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
            row[3] * y_shape)
        cv2.putText(test_frame, str(labels[bbox_index]) + '.....' + str("%.2f" % float(row[4])), (s1 + 50, t1 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(test_frame, (s1, t1), (s2, t2), (0, 255, 0), 2)
        name2=str(name)+'_' + str(datetime.now().timestamp())
        try:
            cv2.imwrite('img/'+name2 + '.jpg', test_frame)
            cv2.imwrite('dataset/'+name2 + '.jpg', frame2_)
        except Exception as e:
            file_error = open("error_anno3.txt", "w")
            file_error.write(e)
            file_error.close()
        # Example usage:
        try:
            create_xml_annotation(name2,class_name,x_shape,y_shape,depth,s1, t1, s2, t2)
        except Exception as e:
            file_error = open("error_anno.txt", "w")
            file_error.write(e)
            file_error.close()




def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def Gesture(userid,frame,stream_url):
    global currentConnections
    connection_dict=currentConnections[str(userid)][str(stream_url)]
    # This function recognizes gestures based on input data
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# Convert frame to RGB color space
    frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results1 = gesture.score_frame(frame) # Perform gesture recognition on the frame
    ges_detection = gesture.plot_boxes(results1, frame) # Plot bounding boxes on the frame
    test = frame
    labels, cord = results1  # Unpack the results into 'labels' and 'cord'

    if (ges_detection != ''):
        connection_dict["gesture"] += 1 # Increment gesture count for the current connection
        if connection_dict["gesture"] == 2 and ges_detection != connection_dict["ges_prev"]:
            connection_dict["res5"] = True
            print('gesture detected')
            connection_dict["ges_prev"] = ges_detection

            connection_dict["gesture"] = 0
            ##################################
            save_img(labels, cord,'ges',test,frame2)
            #####################################

        else:
            connection_dict["res5"]=False
    else:
        connection_dict["res5"] = False
        connection_dict["gesture"] = 0

def Weapon_Tracking(userid,frame,stream_url):
    
    frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    global currentConnections
    connection_dict=currentConnections[str(userid)][str(stream_url)]
    results2 = weapon_tracking.score_frame(frame)
    detection,c1,bbox_index = weapon_tracking.plot_boxes(results2, frame)
    
    
    test = frame
    labels, cord = results2
    if (detection != '') and (c1 != ''):
        connection_dict["counter_W"] += 1
        if connection_dict["counter_W"]==2 and detection != connection_dict["weap_prev"]:
            if connection_dict["zones_dict"]:
                for key, value in connection_dict["zones_dict"].items():
                    print('key',key)
                    print('value',value)
                    x=[]
                    y=[]
                    for i,j in enumerate(value):
                        print('i',i,'j',j)
                        x.append(j[0])
                        y.append(j[1])
                        
                    # print('x',x,'y',y)
                    
                    min_x=min(x)
                    max_x=max(x)
                    min_y=min(y)
                    max_y=max(y)

                    

                    x_shape, y_shape = frame2.shape[1], frame2.shape[0]
                    row = cord[bbox_index]
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    center = ((x1+x2)/2, (y1+y2)/2)
                    if (max_x < center[0] or center[0] < min_x) or (max_y < center[1] or center[1] < min_y):
                        connection_dict["res"] = {'Weapon':'outside_zone'} #c1
                    else:
                        connection_dict["res"] = {'Weapon':'inside '+str(key)}
                        break
                    print('weapon detected....',connection_dict["res"])
                    connection_dict["weap_prev"]=detection
                    connection_dict["detectiontime"] = datetime.now().timestamp()
                    connection_dict["detectiontype"] = "weapon"
                    connection_dict["detectiontype2"].append("weapon")
                    connection_dict["counter_W"] = 0
                    # if (1 in connection_dict["zone_list"]) and (1 in connection_dict["out_list"]):
                    #     x_shape, y_shape = frame2.shape[1], frame2.shape[0]
                    #     row = cord[bbox_index]
                    #     x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    #     center = ((x1+x2)/2, (y1+y2)/2)
                    #     if (connection_dict["max_x"] < center[0] or center[0] < connection_dict["min_x"]) or (connection_dict["max_y"] < center[1] or center[1] < connection_dict["min_y"]):
                    #         connection_dict["res"] = {'Weapon':'outside_zone'} #c1
                    #     else:
                    #         connection_dict["res"] = {'Weapon':'inside_zone'}
                    #     print('weapon detected....',connection_dict["res"])
                    #     connection_dict["weap_prev"]=detection
                    #     connection_dict["detectiontime"] = datetime.now().timestamp()
                    #     connection_dict["detectiontype"] = "weapon"
                    #     connection_dict["detectiontype2"].append("weapon")
                    #     connection_dict["counter_W"] = 0
                    #     #########################3
                    #     save_img(labels, cord,'weapon',c1,test,frame2,bbox_index)
                    # elif (1 in connection_dict["zone_list"]):
                    #     x_shape, y_shape = frame2.shape[1], frame2.shape[0]
                    #     row = cord[bbox_index]
                    #     x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    #     center = ((x1+x2)/2, (y1+y2)/2)
                    #     if not((connection_dict["max_x"] < center[0] or center[0] < connection_dict["min_x"]) or (connection_dict["max_y"] < center[1] or center[1] < connection_dict["min_y"])):
                    #         connection_dict["res"] = {'Weapon':'inside_zone'} #c1
                    #     print('weapon detected....',connection_dict["res"])
                    #     connection_dict["weap_prev"]=detection
                    #     connection_dict["detectiontime"] = datetime.now().timestamp()
                    #     connection_dict["detectiontype"] = "weapon"
                    #     connection_dict["detectiontype2"].append("weapon")
                    #     connection_dict["counter_W"] = 0
                    #     #########################3
                    #     save_img(labels, cord,'weapon',c1,test,frame2,bbox_index)
                    # else:
                    #     x_shape, y_shape = frame2.shape[1], frame2.shape[0]
                    #     row = cord[bbox_index]
                    #     x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    #     center = ((x1+x2)/2, (y1+y2)/2)
                    #     if (connection_dict["max_x"] < center[0] or center[0] < connection_dict["min_x"]) or (connection_dict["max_y"] < center[1] or center[1] < connection_dict["min_y"]):
                    #         connection_dict["res"] = {'Weapon':'outside_zone'} #c1
                    #     print('weapon detected....',connection_dict["res"])
                    #     connection_dict["weap_prev"]=detection
                    #     connection_dict["detectiontime"] = datetime.now().timestamp()
                    #     connection_dict["detectiontype"] = "weapon"
                    #     connection_dict["detectiontype2"].append("weapon")
                    #     connection_dict["counter_W"] = 0
                    #     #########################3
                    save_img(labels, cord,'weapon',c1,test,frame2,bbox_index)

            else:
                connection_dict["res"] = {'Weapon':"None"}
                print('weapon detected....',connection_dict["res"])
                connection_dict["weap_prev"]=detection
                connection_dict["detectiontime"] = datetime.now().timestamp()
                connection_dict["detectiontype"] = "weapon"
                connection_dict["detectiontype2"].append("weapon")
                connection_dict["counter_W"] = 0
                #########################3
                save_img(labels, cord,'weapon',c1,test,frame2,bbox_index)
                ##############################
        else:
            connection_dict["res"]=False

    else:
        connection_dict["res"] = False
        connection_dict["counter_W"] = 0

def Baby_Tracking(userid,frame,stream_url):#,min_x, max_x, min_y, max_y):
    
    frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    global currentConnections
    connection_dict=currentConnections[str(userid)][str(stream_url)]
    results3 = baby_pet.score_frame(frame)
    detection,c,prediction_index = baby_pet.plot_boxes(results3, frame)#,min_x, max_x, min_y, max_y)
    test = frame
    labels, cord = results3
    if (detection != "")and (c != ''):
        connection_dict["counter_B"] += 1
        if (detection != connection_dict["baby_prev"]) and (connection_dict["counter_B"]==3):
            temp_list=list(set(c))   
            if len(temp_list)<2:
                if temp_list[0] == "Baby":
                    pred2='Baby'
                    pred3='Baby'
                else:
                    pred2='Pet'
                    if temp_list[0] == 'Cat':
                        pred3 = 'Cat'
                    else:
                        pred3 = 'Dog'
            else:
                pred2='Baby and Pet'
                pred3=temp_list[0]
            if connection_dict["zones"]:
                for key, value in connection_dict["zones_dict"].items():
                    x=[]
                    y=[]
                    for i,j in enumerate(value):
                        # print('i',i,'j',j)
                        x.append(j[0])
                        y.append(j[1])
                        
                    # print('x',x,'y',y)
                    
                    min_x=min(x)
                    max_x=max(x)
                    min_y=min(y)
                    max_y=max(y)
                    index_number= c.index(pred3)
                    bbox_index=prediction_index[index_number]
                    x_shape, y_shape = frame2.shape[1], frame2.shape[0]
                    row = cord[bbox_index]
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    center = ((x1+x2)/2, (y1+y2)/2)

                    if (max_x < center[0] or center[0] < min_x) or (max_y < center[1] or center[1] < min_y):
                        connection_dict["res2"] = {pred2:'outside_zone'}  
                    else:
                        connection_dict["res2"] = {pred2:'inside '+str(key)}
    
                    print('Baby Runaway')
                    ###############################
                    save_img(labels, cord,'baby',temp_list[0],test,frame2,bbox_index)
                    ################################
                    connection_dict["baby_prev"] = detection
                    connection_dict["detectiontime"] = datetime.now().timestamp()
                    connection_dict["detectiontype"] = "baby"
                    connection_dict["detectiontype2"].append("baby")
                    connection_dict["counter_B"] = 0
            else:
                connection_dict["res2"] = {pred2: 'None'}
                index_number= c.index(pred3)
                bbox_index=prediction_index[index_number]
   
                print('Baby Runaway')
                ###############################
                save_img(labels, cord,'baby',temp_list[0],test,frame2,bbox_index)
                ################################
                connection_dict["baby_prev"] = detection
                connection_dict["detectiontime"] = datetime.now().timestamp()
                connection_dict["detectiontype"] = "baby"
                connection_dict["detectiontype2"].append("baby")
                connection_dict["counter_B"] = 0

        else:
            connection_dict["res2"]=False
    else:

        connection_dict["res2"] = False
        connection_dict["counter_B"] = 0
        # inte1 = 0

def Fire_Tracking(userid,frame,stream_url):
    
    frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    global currentConnections
    connection_dict=currentConnections[str(userid)][str(stream_url)]
    results4 = fire.score_frame(frame)
    detection,co,bbox_index = fire.plot_boxes(results4, frame)
    test = frame
    labels, cord = results4


    if (detection != '') and (co !=''):
        connection_dict["counter_F"] += 1
        if (detection != connection_dict["fire_prev"]) and (connection_dict["counter_F"]==2):
            if connection_dict['zones']:
                for key, value in connection_dict["zones_dict"].items():
                    x=[]
                    y=[]
                    for i,j in enumerate(value):
                        # print('i',i,'j',j)
                        x.append(j[0])
                        y.append(j[1])
                        
                    # print('x',x,'y',y)
                    
                    min_x=min(x)
                    max_x=max(x)
                    min_y=min(y)
                    max_y=max(y)
                    x_shape, y_shape = frame2.shape[1], frame2.shape[0]
                    row = cord[bbox_index]
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    center = ((x1+x2)/2, (y1+y2)/2)
                    if (max_x < center[0] or center[0] < min_x) or (max_y < center[1] or center[1] < min_y):
                        connection_dict["res3"] = {'Fire':'outside_zone'} #c1
                    else:
                        connection_dict["res3"] = {'Fire':'inside '+str(key)}
                
                
                    print('fire alert',connection_dict["res3"])
                    connection_dict["fire_prev"] = detection
                    connection_dict["detectiontime"] = datetime.now().timestamp()
                    connection_dict["detectiontype"] = "fire"
                    connection_dict["detectiontype2"].append("fire")
                    connection_dict["counter_F"] = 0
                    ####################################
                    save_img(labels, cord,'fire',co,test,frame2,bbox_index)
            else:
                connection_dict["res3"] = {'Fire':'None'}
                print('fire alert',connection_dict["res3"])
                connection_dict["fire_prev"] = detection
                connection_dict["detectiontime"] = datetime.now().timestamp()
                connection_dict["detectiontype"] = "fire"
                connection_dict["detectiontype2"].append("fire")
                connection_dict["counter_F"] = 0
                ####################################
                save_img(labels, cord,'fire',co,test,frame2,bbox_index)
            ########################################
        else:
            connection_dict["res3"] = False
    else:
        connection_dict["res3"] = False
        connection_dict["counter_F"] = 0

def Parcel_Tracking(userid,frame,stream_url):
    
    frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    global currentConnections
    connection_dict=currentConnections[str(userid)][str(stream_url)]
    results5 = parcel.score_frame(frame)
    detection,co1,bbox_index = parcel.plot_boxes(results5, frame)
    test = frame
    labels, cord = results5

    if (detection != '') and (co1 !=''):
        connection_dict["counter_P"] += 1
        if (detection != connection_dict["parcel_prev"]) and (connection_dict["counter_P"]==3):
            if connection_dict['zones']:
                for key, value in connection_dict["zones_dict"].items():
                    x=[]
                    y=[]
                    for i,j in enumerate(value):
                        # print('i',i,'j',j)
                        x.append(j[0])
                        y.append(j[1])
                        
                    # print('x',x,'y',y)
                    
                    min_x=min(x)
                    max_x=max(x)
                    min_y=min(y)
                    max_y=max(y)
                    x_shape, y_shape = frame2.shape[1], frame2.shape[0]
                    row = cord[bbox_index]
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    center = ((x1+x2)/2, (y1+y2)/2)
                    if (max_x < center[0] or center[0] < min_x) or (max_y < center[1] or center[1] < min_y):
                        connection_dict["res1"] = {'Parcel':'outside_zone'} #c1
                    else:
                        connection_dict["res1"] = {'Parcel':'inside '+str(key)}
            
                    connection_dict["parcel_prev"] = detection
                    connection_dict["detectiontime"] = datetime.now().timestamp()
                    connection_dict["detectiontype"] = "parcel"
                    connection_dict["detectiontype2"].append("parcel")
                    connection_dict["counter_P"] = 0
                    ###########################################
                    save_img(labels, cord,'parcel',co1,test,frame2,bbox_index)
                #############################################
            else:
                connection_dict["res1"] = {'Parcel':'None'}
                # connection_dict["res1"] = 'Parcel' #co
                connection_dict["parcel_prev"] = detection
                connection_dict["detectiontime"] = datetime.now().timestamp()
                connection_dict["detectiontype"] = "parcel"
                connection_dict["detectiontype2"].append("parcel")
                connection_dict["counter_P"] = 0
                ###########################################
                save_img(labels, cord,'parcel',co1,test,frame2,bbox_index)
        else:
            connection_dict["res1"] = False

    else:

        connection_dict["res1"] = False
        connection_dict["counter_P"] = 0

def Dog_poop_Tracking(userid,frame,stream_url):
    
    frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    global currentConnections
    connection_dict=currentConnections[str(userid)][str(stream_url)]

    

    results6 = dog_poop.score_frame(frame)
    detection,co2,bbox_index = dog_poop.plot_boxes(results6, frame)
    test = frame
    labels, cord = results6

    if (detection != '') and (co2 !=''):
        connection_dict["counter_D"] += 1
        if connection_dict["counter_D"] == 2 and detection != connection_dict["dog_prev"]:
            if connection_dict['zones']:
                for key, value in connection_dict["zones_dict"].items():
                    x=[]
                    y=[]
                    for i,j in enumerate(value):
                        # print('i',i,'j',j)
                        x.append(j[0])
                        y.append(j[1])
                        
                    # print('x',x,'y',y)
                    
                    min_x=min(x)
                    max_x=max(x)
                    min_y=min(y)
                    max_y=max(y)
                    x_shape, y_shape = frame2.shape[1], frame2.shape[0]
                    row = cord[bbox_index]
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    center = ((x1+x2)/2, (y1+y2)/2)
                    if (max_x < center[0] or center[0] < min_x) or (max_y < center[1] or center[1] < min_y):
                        connection_dict["res4"] = {'Dog Poop':'outside_zone'} #c1
                    else:
                        connection_dict["res4"] = {'Dog Poop':'inside '+str(key)}       
                #############################################
                    connection_dict["dog_prev"] = detection 
                    connection_dict["detectiontime"] = datetime.now().timestamp()
                    connection_dict["detectiontype"] = "dog"
                    connection_dict["detectiontype2"].append("dog")
                    connection_dict["counter_D"] = 0
                    ################################
                    save_img(labels, cord,'dog_poop',co2,test,frame2,bbox_index)
            #################################
            else:
                connection_dict["res4"] = {'Dog Poop':'None'} 
                connection_dict["dog_prev"] = detection 
                connection_dict["detectiontime"] = datetime.now().timestamp()
                connection_dict["detectiontype"] = "dog"
                connection_dict["detectiontype2"].append("dog")
                connection_dict["counter_D"] = 0
                ################################
                save_img(labels, cord,'dog_poop',co2,test,frame2,bbox_index)

        else:
            connection_dict["res4"] = False
    else:

        connection_dict["res4"] = False
        connection_dict["counter_D"] = 0

def Animal_Tracking(userid,frame,stream_url):
    
    
    frame2 = frame#cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    global currentConnections,results7
    connection_dict=currentConnections[str(userid)][str(stream_url)]
    connection_dict["person_count"]=0

    # global  res6,ani_prev,len_of_list6,currentConnections
    vehicles = ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
    animals = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
    
    results7 = veh_and_ani.score_frame(frame)
    detection,pred,person_count,bbox_index = veh_and_ani.plot_boxes(results7, frame)
    connection_dict["person_count"]=person_count
    test = frame
    labels, cord = results7
    co3=pred
    if (detection != '') and (pred != ''):
        connection_dict["counter_A"] += 1

        if connection_dict["counter_A"] == 1 and detection != connection_dict["ani_prev"]:
            # print('******************')
            if pred in vehicles:
                pred = 'Vehicle'
            if pred in animals:
                pred= 'Animal'
            if pred == 'person':
                pred='Person'
            if connection_dict["zones"]:
                for key, value in connection_dict["zones_dict"].items():
                    x=[]
                    y=[]
                    for i,j in enumerate(value):
                        # print('i',i,'j',j)
                        x.append(j[0])
                        y.append(j[1])
                        
                    # print('x',x,'y',y)
                    
                    min_x=min(x)
                    max_x=max(x)
                    min_y=min(y)
                    max_y=max(y)
                    x_shape, y_shape = frame2.shape[1], frame2.shape[0]
                    row = cord[bbox_index]
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    center = ((x1+x2)/2, (y1+y2)/2)
                    if (max_x < center[0] or center[0] < min_x) or (max_y < center[1] or center[1] < min_y):
                        connection_dict["res6"] = {pred:'outside_zone'} #c1
                    else:
                        connection_dict["res6"] = {pred:'inside '+str(key)}  
                    # connection_dict["res6"] = pred
                    print('Animal and vehicle.....',pred)
                    if pred != 'Person':
                        save_img(labels, cord,'ani_'+str(pred),co3,test,frame2,bbox_index)
                    connection_dict["ani_prev"] = detection
                    connection_dict["detectiontime"] = datetime.now().timestamp()
                    connection_dict["detectiontype"] = pred
                    connection_dict["detectiontype2"].append(pred)
                    connection_dict["counter_A"] = 0
            else:
                connection_dict["res6"] = {pred:'inside_zone'}
                print('Animal and vehicle.....',pred)
                if pred != 'Person':
                    save_img(labels, cord,'ani_'+str(pred),co3,test,frame2,bbox_index)
                connection_dict["ani_prev"] = detection
                connection_dict["detectiontime"] = datetime.now().timestamp()
                connection_dict["detectiontype"] = pred
                connection_dict["detectiontype2"].append(pred)
                connection_dict["counter_A"] = 0

        else:
            connection_dict["res6"] = False
    else:

        connection_dict["res6"] = False
        connection_dict["counter_A"] = 0

def Window_Tracking(userid,frame,stream_url):
    
    frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    global currentConnections,results8
    connection_dict=currentConnections[str(userid)][str(stream_url)]
    results8 = window.score_frame(frame)
    co8,bbox_index = window.plot_boxes(results8, frame)
    test = frame
    labels, cord = results8
    n = len(labels)
    if (co8 !=''):
        connection_dict["counter_W"] += 1
        if (connection_dict["counter_W"]==1):
            if connection_dict["zones"]:
                for key, value in connection_dict["zones_dict"].items():
                    x=[]
                    y=[]
                    for i,j in enumerate(value):
                        # print('i',i,'j',j)
                        x.append(j[0])
                        y.append(j[1])
                        
                    # print('x',x,'y',y)
                    
                    min_x=min(x)
                    max_x=max(x)
                    min_y=min(y)
                    max_y=max(y)
                    x_shape, y_shape = frame2.shape[1], frame2.shape[0]
            
                    row = cord[bbox_index]
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    center = ((x1+x2)/2, (y1+y2)/2)
                    if (max_x < center[0] or center[0] < min_x) or (max_y < center[1] or center[1] < min_y):
                        connection_dict["res7"] = {'Window':'outside_zone'} #c1
                    else:
                        connection_dict["res7"] = {'Window':'inside '+str(key)}

                    print('!!!!! window detected')
            
                    connection_dict["detectiontime"] = datetime.now().timestamp()
                    connection_dict["detectiontype"] = "window"
                    connection_dict["eaves"].append('window')# here word window is generalized term for "Window" or "Door"
                    connection_dict["counter_W"] = 0
            else:
                connection_dict["res7"] = {'Window':'None'}
                print('!!!!! window detected')
               
                connection_dict["detectiontime"] = datetime.now().timestamp()
                connection_dict["detectiontype"] = "window"
                connection_dict["eaves"].append('window')# here word window is generalized term for "Window" or "Door"
                connection_dict["counter_W"] = 0

            
        else:
            connection_dict["res7"] = False
    else:
        connection_dict["res7"] = False
        connection_dict["counter_W"] = 0
def Jumping(userid,stream_url,frame_original,device_n):
    print(']]]]]]]')
    global currentConnections
    connection_dict=currentConnections[str(userid)][str(stream_url)]
    jump=jumping.predict_single_action(connection_dict["frames"])
    if jump == True:
        result_jump=send_notifcation(frame_original,userid,' someone jumped from the wall',device_n,'jumping')
        # send_notifcation(frame_original,userid,'Weapon Alert ('+connection_dict["res"]['Weapon']+')',device_n,connection_dict["res"]['Weapon'])
        print('jump notification sent')
        jump==False
    connection_dict["frames"]=[]

# functions that work for each stream in thread
def livestream(stream_url,userid,device_n):
    
    global currentConnections
    cap = cv2.VideoCapture(stream_url)
    print('/////////////tttt',cap.isOpened())

    #stream_url = "http://167.86.110.200:5080/LiveApp/streams/main-cam-5cdb5fdacadda6e5.m3u8"
    
    max_retries = 3  # Maximum number of retry attempts
    retry_delay = 5  # Delay between retry attempts in seconds

    retry_count = 0
    connected = False

    
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
    # cap.set(cv2.CAP_PROP_FPS,10)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # # print("Default FPS:", fps)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    counter=0

    print('/////////////kkk',cap.isOpened())
    #if user already exists and reconnects hen its stream start time will be update by current
    if (userid in currentConnections) and (str(stream_url) in currentConnections[str(userid)]):#(userid in currentConnections)
        # if str(stream_url) in currentConnections[str(userid)]:
        print('...if condition satisfied....')
        
        currentConnections[str(userid)][str(stream_url)]["stream_start_time"] = datetime.now().strftime("%H:%M:%S")
    count = 5
    # if stream is connected it will try to connect it 3 times 
    while(not cap.isOpened()):
        count = count -1
        cap = None
        # Create a VideoCapture object
        cap = cv2.VideoCapture()

        # Set the timeout value in milliseconds
        timeout = 10000  # 10 seconds

        # Set the timeout property on the VideoCapture object
        #cap.set(cv2.CAP_PROP_TIMEOUT, timeout)

        # Open the HLS stream
        cap.open(stream_url)
        print('......checking stream',cap.isOpened())
        if cap.isOpened():
            break
        
        if count == 0:
            break
        
        #time.sleep(5)


    #if stream gets connected this loop starts which contains all alerts
    while(cap.isOpened()):
        start_time = time.time()
        
        
        # try:
        ret, frame_original = cap.read()
            # print('/////',ret)
        # except Exception as e:
        #     print('.......',e)
        #     break
        frame_original=cv2.resize(frame_original, (720,1280))#960*528#640*352#720*1280 our current


        # frame_original = cv2.rotate(frame_original, cv2.ROTATE_90_COUNTERCLOCKWISE)#nasir
        # frame_original=cv2.resize(frame_original, (720,1280))#960*528#640*352#720*1280
        # frame_original = cv2.rotate(frame_original, cv2.ROTATE_90_CLOCKWISE)#bangash
        


        if not ret:
            break
        try:
           # this block is to check if audio and video both are coming from videCapture if not than wait for 5 iteration otherwise send notification that stream is stopped
            
            try:
                frame_notification=frame_original.copy()
            except Exception as e:
               
                print('EXCEPTION',e)
                currentConnections[str(userid)][str(stream_url)]["frame_exception"] +=1
                if currentConnections[str(userid)][str(stream_url)]["frame_exception"] >5:
                    currentConnections[str(userid)][str(stream_url)]["frame_exception"]=0
                    cap = cv2.VideoCapture(stream_url)
                    if not cap.isOpened():

                        result_stream=send_notifcation(None,userid,'stream stopped',device_n)
                        print('Restart doorbell',result_stream)
                        break
           
                continue
           
            # cv2.imwrite("temp/temp_"+str(userid)+".jpeg",frame_original)

            #if user exist in dict than increase the counter and save frames
            if (userid in currentConnections) and (str(stream_url) in currentConnections[str(userid)]):#(userid in currentConnections):
                # if str(stream_url) in currentConnections[str(userid)]:
                connection_dict=currentConnections[str(userid)][str(stream_url)]
                connection_dict["counter"]+=1
                if connection_dict["counter"] %30 ==0:
                    # print('uncomment below line')
                    connection_dict["frames"].append(frame_original)

            #this will work first time user connects to the system
            else:
                #dict responsible for maintaining the previous info of a particular user
                currentConnections[str(userid)]= {str(stream_url): {
                    "stream_status":True,
                    "width":0,
                    "height":0,
                    "intruder":False,
                    "frame_exception":0,
                    "userid":str(userid),
                    # "stream_url":str(stream_url),
                    "stream_stop":0,
                    "stream_start_time":datetime.now().strftime("%H:%M:%S"),
                    "old_zone":0,
                    "zones":False,#[[20,10],[150,10],[250,10],[20,500],[150,500],[250,500],[10,200],[100,200]], #top_l, top_m, top_r, bottom_l, bottom_m, bottom_r, left_m, right_m
                    "zone_list":[1,2,3,4,5,6,7],# weapon,baby,animal&vehicle,parcel,fire,eaves,dog_poop
                    "out_list":[2,3,4],
                    "zones_dict":{},
                    "notifications":None,
                    "previous_frame":None,
                    "check":False,
                    "person_count":0,
                    "min_x":0,
                    "min_y":0,
                    "max_x":0,
                    "max_y":0,
                    "res":'',
                    "res1":'',
                    "res2":'',
                    "res3":'',
                    "res4":'',
                    "res5":'',
                    "res6":'',
                    "res7":'',
                    "weap_prev":'',
                    "baby_prev":'',
                    "fire_prev":'',
                    "parcel_prev":'',
                    "dog_prev":'',
                    "ani_prev":'',
                    "ges_prev":'',
                    # "window_prev":'',
                    "counter_W":0,
                    "counter_B": 0,
                    "counter_F": 0,
                    "counter_P":0,
                    "counter_D": 0,
                    "counter_A": 0,
                    "counter_W": 0,
                    "counter":0,
                    "gesture":0,
                    "motion_var":0,
                    "motion_start":0,
                    'motion_start_time':0,
                    'motion_end_time':0,
                    "motion_":False,
                    "motion_end":0,
                    "uni_detection":'motion',
                    "detected":False,
                    "detectiontime":time.time(),
                    "detectiontype":"",
                    "detectiontype2":[],
                    "frames": [],
                    "backup":[],
                    "eaves":[],
                    "notification_ids":[],
                    "notification_time":[]

                }}
                with open('user_info.json', 'r') as json_file:
                    data = json.load(json_file)
                if userid in data and stream_url in data[userid]:
                    zones_dict_value = data[userid][stream_url]['zones_dict']
                    print("Zones Dict:", zones_dict_value)
                    currentConnections[str(userid)][str(stream_url)]["zones_dict"]=zones_dict_value
                #all the info for e.g time &date is sent to db when stream starts
                send_alert_timer(currentConnections[str(userid)][str(stream_url)]["userid"],stream_url,start_time=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),type="started_stream")
            #############################
            # if len(currentConnections[str(userid)][str(stream_url)]["frames"]) == 20:
            #     t_jump = Thread(target=Jumping, args=(str(userid),str(stream_url),frame_original,device_n))
            #     t_jump.start()
            #     t_jump.join()
            ######################################
            connection_dict=currentConnections[str(userid)][str(stream_url)]
            # cv2.circle(frame_original, connection_dict['zones'][0], 5, (0,0,0), -1)
            # cv2.circle(frame_original, connection_dict['zones'][1], 5, (0,0,0), -1)
            # cv2.circle(frame_original, connection_dict['zones'][2], 5, (0,0,0), -1)
            # cv2.circle(frame_original, connection_dict['zones'][3], 5, (0,0,0), -1)
            # cv2.circle(frame_original, connection_dict['zones'][4], 5, (0,0,0), -1)
            # cv2.circle(frame_original, connection_dict['zones'][5], 5, (0,0,0), -1)
            # cv2.circle(frame_original, connection_dict['zones'][6], 5, (0,0,0), -1)
            # cv2.circle(frame_original, connection_dict['zones'][7], 5, (0,0,0), -1)
            ##############zone################
            # if connection_dict['zones_dict']:
            #     # if connection_dict['old_zone'] != connection_dict['zones']:
             
            #     x=[]
            #     y=[]
            #     for i,j in enumerate(connection_dict['zones']):
            #         print('i',i,'j',j)
            #         x.append(j[0])
            #         y.append(j[1])
                    
            #     print('x',x,'y',y)
                
            #     connection_dict['min_x']=min(x)
            #     connection_dict['max_x']=max(x)
            #     connection_dict['min_y']=min(y)
            #     connection_dict['max_y']=max(y)

            #     connection_dict['old_zone']=connection_dict['zones']
            ###########zone###################
            frame_height, frame_width = frame_original.shape[:2]
            # print('width',frame_width,'....height',frame_height)
            
            if int(connection_dict["height"])==0 or (frame_height == int(connection_dict["height"]) and frame_width == int(connection_dict["width"])):
                connection_dict["width"]=frame_width
                connection_dict["height"]=frame_height
            #after 30 iterations motion detection is checked
                if int(connection_dict["counter"])%90==0 and int(connection_dict["counter"]) !=0:#30
                    cv2.imwrite("temp/temp_"+str(userid)+".jpeg",frame_original)
                    
                   ############ frame mismatch motion detection correction ################
                    # target_size = (640, 640)
                    # frame_original = resize_and_pad_image(frame_original, target_size)
                    ############ frame mismatch motion detection correction ################

                    ########################### Activity zone #######################
                    frame=frame_original.copy()
                    # min_x, max_x, min_y, max_y = get_region(frame,connection_dict["zones"],frame_height,frame_width)
                    ########################### Activity zone #######################

                    if connection_dict["check"] == False:
                        #converting the frame to rgb than to gray  than applying guassian blurr
                        try:
                            print('//////// here')
                            connection_dict["previous_frame"] = frame
                            img_brg1 = np.array(connection_dict["previous_frame"])
                            img_rgb1 = cv2.cvtColor(src=img_brg1, code=cv2.COLOR_BGR2RGB)

                            # 2. Prepare image; grayscale and blur
                            connection_dict["previous_frame"] = cv2.cvtColor(img_rgb1, cv2.COLOR_BGR2GRAY)
                            connection_dict["previous_frame"] = cv2.GaussianBlur(src=connection_dict["previous_frame"], ksize=(5,5), sigmaX=0)
                        except:
                            print("nope")
                    #sending the current and previous frame to motion_detect file
                
                    text,prev_frame = motion.motion_detect(frame,connection_dict["previous_frame"])
          
                    connection_dict["previous_frame"] = prev_frame
                    #if motion is detected the motion variable counter is initialized
                    if text == 'DANGER':
                        connection_dict["motion_var"]=0
                        
                        
                        if connection_dict["motion_"] != True:
                            
                            connection_dict["motion_start"]=connection_dict["counter"]
                            connection_dict["motion_start_time"]=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            send_alert_timer(connection_dict["userid"],stream_url,connection_dict["motion_start_time"],
                                                 type='Motion_Started')
                    


                        print('............. Motion Detected ...............',userid)
                    


                        ########### blurr check ##########
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        fm = variance_of_laplacian(gray)

                        #if blurriness is lower than the threshold we will start all the threads
                        if fm > 0.0: #100
                            if int(connection_dict["counter"])%90==0:#60
                                im=frame
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                print('*******')
                                # t2 = Thread(target=Weapon_Tracking, args=(str(userid),frame,str(stream_url)))#,min_x, max_x, min_y, max_y))
                                # t3 = Thread(target=Baby_Tracking, args=(str(userid),frame,str(stream_url)))#,min_x, max_x, min_y, max_y))
                                # t4 = Thread(target=Fire_Tracking, args=(str(userid), frame,str(stream_url)))#,min_x, max_x, min_y, max_y))
                                # t5 = Thread(target=Dog_poop_Tracking, args=(str(userid), frame,str(stream_url)))#,min_x, max_x, min_y, max_y))
                                # t6 = Thread(target=Animal_Tracking, args=(str(userid), im,str(stream_url)))#,min_x, max_x, min_y, max_y))
                                # t7 = Thread(target=Parcel_Tracking, args=(str(userid), frame,str(stream_url)))#,min_x, max_x, min_y, max_y))
                                # t8 = Thread(target=Window_Tracking, args=(str(userid), frame,str(stream_url)))#,min_x, max_x, min_y, max_y))



                                # t2.start()
                                # t3.start()
                                # t4.start()
                                # t5.start()
                                # t6.start()
                                # t7.start()
                                # t8.start()

                                # t1.join()
                                # t2.join()
                                # t4.join()
                                # t5.join()

                                # t3.join()
                                # t6.join()
                                # t8.join()
                                Weapon_Tracking(str(userid),frame,str(stream_url))
                                Baby_Tracking(str(userid),frame,str(stream_url))
                                # Fire_Tracking(str(userid),frame,str(stream_url))
                                # Dog_poop_Tracking(str(userid),frame,str(stream_url))
                                # Animal_Tracking(str(userid),frame,str(stream_url))
                                # print('.....*********.......successful.....*********.....')
                                #these are the checks in which we are checking if theres any detection. we send notification with the text according to the detection
                                if connection_dict["res"]:
                                    print('resltttttt',connection_dict["res"]['Weapon'])
                                    result=send_notifcation(frame_original,userid,'Weapon Alert ('+connection_dict["res"]['Weapon']+')',device_n,connection_dict["res"]['Weapon'])
                                    connection_dict["notification_ids"].append(result)
                                    connection_dict["notification_time"].append(datetime.now().strftime("%H:%M:%S"))
                                    send_alert_timer(connection_dict["userid"],stream_url,connection_dict
                                                ["motion_start_time"],type='Weapon',notification_id=result,end_time=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                                    print('notification id',connection_dict["notification_ids"])
                                    print('weapon notification sent',result)

                                if connection_dict["res1"]:
                                    
                                    result1=send_notifcation(frame_original,userid,'Parcel left at porch ('+connection_dict["res1"]['Parcel']+')',device_n,connection_dict["res1"]['Parcel'])
                                    connection_dict["notification_ids"].append(result1)
                                    connection_dict["notification_time"].append(datetime.now().strftime("%H:%M:%S"))
                                    send_alert_timer(connection_dict["userid"],stream_url,connection_dict
                                                ["motion_start_time"],type='Parcel',notification_id=result1,end_time=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                                    print('blocked notification of parcel on my side')
                                    

                                if connection_dict["res2"]:
                             
                                    if connection_dict["person_count"]<2:
                                        # print('........',connection_dict["res2"]+' Spotted Alone')
                                        print(connection_dict["res2"])
                                        result2=send_notifcation(frame_original,userid,list(connection_dict["res2"].keys())[0]+' Spotted Alone ('+list(connection_dict["res2"].values())[0]+')',device_n,list(connection_dict["res2"].values())[0])
                                        connection_dict["notification_ids"].append(result2)
                                        connection_dict["notification_time"].append(datetime.now().strftime("%H:%M:%S"))
                                        send_alert_timer(connection_dict["userid"],stream_url,connection_dict
                                                ["motion_start_time"],type=list(connection_dict["res2"].keys())[0],notification_id=result2,
                                                end_time=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                                        print('baby_pet notification sent',result2)


                                if connection_dict["res3"]:
                                    print('entered')
                                    
                                    result3=send_notifcation(frame_original,userid,'Fire detected in front of the porch ('+connection_dict["res3"]['Fire']+')',device_n,connection_dict["res3"]['Fire'])
                                    connection_dict["notification_ids"].append(result3)
                                    connection_dict["notification_time"].append(datetime.now().strftime("%H:%M:%S"))
                                    send_alert_timer(connection_dict["userid"],stream_url,connection_dict
                                                ["motion_start_time"],type='Fire',notification_id=result3,
                                                end_time=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                                    print('fire notification sent',result3)


                                if connection_dict["res4"] :
                                    result5=send_notifcation(frame_original,userid,'Dog poop detected ('+connection_dict["res4"]['Dog Poop']+')',device_n,connection_dict["res4"]['Dog Poop'])
                                    connection_dict["notification_ids"].append(result5)
                                    connection_dict["notification_time"].append(datetime.now().strftime("%H:%M:%S"))
                                    send_alert_timer(connection_dict["userid"],stream_url,connection_dict
                                                ["motion_start_time"],type='Dog Poop',notification_id=result5,
                                                end_time=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                                    print('blocked notification of dog poop on my side')


                          
                                        ############################ intruder #################
                                if connection_dict["res6"]:  
                                    if connection_dict["intruder"] == False:  
                                        if 'Person' not in connection_dict["res6"].keys():
                                            result4=send_notifcation(frame_original,userid,list(connection_dict["res6"].keys())[0]+' detected outside the house ('+list(connection_dict["res6"].values())[0]+')',device_n,list(connection_dict["res6"].values())[0])
                                            connection_dict["notification_ids"].append(result4)
                                            connection_dict["notification_time"].append(datetime.now().strftime("%H:%M:%S"))
                                            send_alert_timer(connection_dict["userid"],stream_url,connection_dict
                                                ["motion_start_time"],type=list(connection_dict["res6"].keys())[0],notification_id=result4,
                                                end_time=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                                            print('ani_veh notification sent',result4)
                                    else:
                                        if 'Person' in connection_dict["res6"].keys():
                                            result4=send_notifcation(frame_original,userid,'Intruder Alert ('+list(connection_dict["res6"].values())[0]+')',list(connection_dict["res6"].values())[0])
                                            connection_dict["notification_ids"].append(result4)
                                            connection_dict["notification_time"].append(datetime.now().strftime("%H:%M:%S"))
                                            send_alert_timer(connection_dict["userid"],stream_url,connection_dict
                                                ["motion_start_time"],type= 'Intruder',notification_id=result4,
                                                end_time=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                                            print('ani_veh notification sent',result4)
                                        else:
                                            result4=send_notifcation(frame_original,userid,connection_dict["res6"].keys()+' detected outside the house ('+list(connection_dict["res6"].values())[0]+')',device_n,list(connection_dict["res6"].values())[0])
                                            connection_dict["notification_ids"].append(result4)
                                            connection_dict["notification_time"].append(datetime.now().strftime("%H:%M:%S"))
                                            send_alert_timer(connection_dict["userid"],stream_url,connection_dict
                                                ["motion_start_time"],type=connection_dict["res6"].keys(),notification_id=result4,
                                                end_time=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                                            print('ani_veh notification sent',result4)

                                if connection_dict["res7"]:
                                    # t6.join()
                                    if connection_dict["res6"]:
                                        print('atleast inside')

                                        if 'Person' in connection_dict["res6"].keys():
                                            box1=(results7[1][0][0:4])#Person
                                            for j in range(len(results8[1])):
                                                box2=(results8[1][j][0:4])#window
                                                if box1[0] < box2[2] and box1[2] > box2[0] and box1[1] < box2[3] and box1[3] > box2[1]:
                                                    
                                                    cv2.imwrite('img/eaves_'+str(datetime.now().timestamp())+'.jpg',cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                                                    result5=send_notifcation(frame_original,userid,'Eavesdropper detected ('+list(connection_dict["res7"].values())[0]+')',device_n,list(connection_dict["res7"].values())[0])
                                                    connection_dict["notification_ids"].append(result5)
                                                    connection_dict["notification_time"].append(datetime.now().strftime("%H:%M:%S"))
                                                    send_alert_timer(connection_dict["userid"],stream_url,connection_dict
                                                        ["motion_start_time"],type='Eavesdropper',notification_id=result5,
                                                        end_time=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                                                    print('Eavesdropping',result5)
    
                                if len(connection_dict["detectiontype2"]) != 0:
                                    all_dets = Counter(connection_dict["detectiontype2"])
                                    all_dets=all_dets.most_common()
                                    most_occur=all_dets[0][0]
                                    if len(all_dets)>1 and ((most_occur =='Person') or (most_occur == "") or (most_occur =='window')):
                                        most_occur=all_dets[1][0]
                        
                                    connection_dict["uni_detection"]=most_occur
                                # frame_start=0
                            connection_dict["motion_"] =True
                        else:
                            print('!!!!!!!!this was blurr frame!!!!!!!!!!',fm)
                    #if motion ends we wait for 15 iterations and then check how many notifications are sent and what were there ids and time and this info to db
                    else:
                        connection_dict["motion_var"]+=1
            
                        if connection_dict["motion_start"] != 0 and connection_dict["motion_var"] > 5:#15
                        
                            connection_dict["motion_end"]=connection_dict["counter"]
                            connection_dict["motion_end_time"]=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            try:
                                send_alert_timer(connection_dict["userid"],stream_url,end_time=connection_dict["motion_end_time"],
                                                 type='Motion_Ended')
                            except:
                                print('alert timer issue')

                            #if no otion is detected after 15 iterations we initialize all keys with null values
                            connection_dict["notification_ids"]=[]
                            connection_dict["notification_time"]=[]
                            connection_dict["detected"] = False
                            connection_dict["motion_var"]=0
                            connection_dict["motion_start"] = 0
                            connection_dict["motion_"] =False
                            connection_dict["detectiontype2"]=[]


                        if connection_dict["motion_var"] > 5:#15
                            connection_dict["uni_detection"]='motion'
                            
                    connection_dict["check"]=True
                connection_dict["counter"]+=1
            else:
                connection_dict["width"]=frame_width
                connection_dict["height"]=frame_height
                connection_dict["previous_frame"] = frame_original.copy()
                img_brg1 = np.array(connection_dict["previous_frame"])
                img_rgb1 = cv2.cvtColor(src=img_brg1, code=cv2.COLOR_BGR2RGB)

                # 2. Prepare image; grayscale and blur
                connection_dict["previous_frame"] = cv2.cvtColor(img_rgb1, cv2.COLOR_BGR2GRAY)
                connection_dict["previous_frame"] = cv2.GaussianBlur(src=connection_dict["previous_frame"], ksize=(5,5), sigmaX=0)
                print('hereeee')
                continue
        
       
        except Exception as e:
            traceback.print_exc()
            print('problem',e)
            continue
        
         
            # if userid in currentConnections:
            #     connection_dict["stream_stop"]+=1
            #     if connection_dict["stream_stop"]==3:
            #         connection_dict["stream_stop"]=0
            #         break
            # else:
            #     continue
        end_time = time.time()
        elapsed_time = end_time - start_time

        
        # if elapsed_time > 0.001:  # You can adjust the threshold as needed
        #     print('fps',1 / elapsed_time)
    cap.release()
    if (userid in currentConnections) and (str(stream_url) in currentConnections[str(userid)]):#userid in currentConnections:
        # if str(stream_url) in currentConnections[str(userid)]:
        connection_dict["stream_status"]=False
        # print('.......stream off ...........')
    
    # print(currentConnections)

import subprocess



# Send Notification Directly to Admin #
def send_notifcation(frame3,userid,message,device_name,zone_status=None):#----
    aiurl = "https://beta.irvinei.com/api/api/send-notification"

    if frame3 is None:
        ##############################
        frame4=cv2.imread('sample.jpg')
        
        success, encoded_frame2 = cv2.imencode('.jpg', frame4)
        filename = str(time.time()) + '_' + str(random.randint(0, 999999)) + ".jpg"
        files = {
            "image": (filename, io.BytesIO(encoded_frame2).getvalue(), "image/jpeg")
        }

        body = {"userid": userid,"title":"Emergency", "message": message,"device_name":device_name}
        getdata = requests.post(aiurl, data=body, files=files)
    else:
        success, encoded_frame = cv2.imencode('.jpg', frame3)
        filename = str(time.time()) + '_' + str(random.randint(0, 999999)) + ".jpg"
        body = {"userid": userid, "message": message,"device_name":device_name}
        files = {
            "image": (filename, io.BytesIO(encoded_frame).getvalue(), "image/jpeg")
        }
        
        getdata = requests.post(aiurl, data=body, files=files)
        

    file_error = open("error_file.txt", "w")
    file_error.write(getdata.text)
    file_error.close()

    try:
        
        parsed_number = int(getdata.text)

        return parsed_number

    except ValueError:
        print("Exception")
        send_notifcation(frame3,userid,message,device_name)


# Send Notification Directly to Admin #
def send_notifcation_after_face(frame,userid):
    print('****n*****')
    faceurl = "http://20.227.167.170:1000/uploader"
    frame = cv2.imencode('.jpg', frame)[1].tostring()
    message =str(time.time())+'_'+str(random.randint(0,999999))+".jpg"
    payload = {'filename': message,'userid': userid}
    files = {'frame': (message, frame)}
    getdata = requests.post(faceurl, data=payload, files=files)
    print(getdata.text)
    return getdata.text

def send_alert_timer(userid,stream_url,start_time=0,end_time=0,type=0,notification_id=0,notification_time=0):
    myurl = "https://beta.irvinei.com/api/api/save-alert"

    if type == "started_stream":
        body = {"userid":userid,"starttime":start_time,"stream_url":stream_url,"alerttype":type,"notification_id":notification_id,"notification_time":notification_time}
    else:
        body = {"userid":userid,"starttime":start_time,"endtime":end_time,"alerttype":type,"stream_url":stream_url,"notification_id":notification_id,"notification_time":notification_time}

    getdata = requests.post(myurl,data=body)#,headers={'Content-Type': 'multipart/form-data'})
    print('notification_id sent',getdata.text, 'notification_id',notification_id)
    file_error = open("error_file.txt", "w")
    file_error.write(getdata.text)
    file_error.close()
    return getdata.text



def readimage(path):
    count = os.stat(path).st_size / 2
    with open(path, "rb") as f:
        return bytearray(f.read())


app = Flask(__name__)
currentConnections = {}
start_time=datetime.now().timestamp()


from pathlib import Path
FILE_TEMPLATE = '''
<a href="/{video_file}">{video_file}</a>
'''
BASE_DIR = Path('videos')
CHUNK_SIZE = 2 ** 25

class RegexConverter(BaseConverter):
    def __init__(self, url_map, *items):
        super(RegexConverter, self).__init__(url_map)
        self.regex = items[0]
app.url_map.converters['regex'] = RegexConverter


def get_chunk(full_path, byte1=None, byte2=None):
    if not os.path.isfile(full_path):
        raise OSError('no such file: {}'.format(full_path))
    file_size = os.stat(full_path).st_size
    start = 0
    if byte1 < file_size:
        start = byte1
    length = file_size - start
    if byte2:
        length = byte2 + 1 - byte1
    if length > CHUNK_SIZE:
        length = CHUNK_SIZE
    with open(full_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(length)
    return chunk, start, length, file_size


def get_byte_interval(request):
    range_header = request.headers.get('Range', None)
    byte1, byte2 = 0, None
    if range_header:
        match = re.search(r'(\d+)-(\d*)', range_header)
        groups = match.groups()

        if groups[0]:
            byte1 = int(groups[0])
        if groups[1]:
            byte2 = int(groups[1])
    return byte1, byte2


@app.route('/all-videos')
def index():

    user = request.args.get('userid')
    list = os.listdir("videos/{0}".format(user))
    return json.dumps(list)

@app.route('/<regex(".*\.mp4"):video_file>')
def get_file_mp4(video_file):
    byte1, byte2 = get_byte_interval(request)
    chunk, start, length, file_size = get_chunk(
        BASE_DIR / video_file,
        byte1, byte2)
    resp = Response(chunk, 206, mimetype='video/mp4',
                    content_type='video/mp4', direct_passthrough=True)
    resp.headers.add('Content-Range', 'bytes {0}-{1}/{2}'.format(start, start + length - 1, file_size))
    return resp

@app.route('/<regex(".*\.flv"):video_file>')
def get_file_flv(video_file):
    byte1, byte2 = get_byte_interval(request)

    chunk, start, length, file_size = get_chunk(
        BASE_DIR / video_file,
        byte1, byte2)
    resp = Response(chunk, 206, mimetype='video/mp4',
                    content_type='video/mp4', direct_passthrough=True)
    resp.headers.add('Content-Range', 'bytes {0}-{1}/{2}'.format(start, start + length - 1, file_size))
    return resp

# @app.route('/intruder', methods=['POST'])
# def Intruder(userid, stream_url):
#     connection_dict["intruder"]=True
@app.route('/get_base64', methods=['POST'])
def img_base():
    url=request.form['url']
    # print('url',url)
    cap2=cv2.VideoCapture(url)
    ret,frame_=cap2.read()
    retval, buffer = cv2.imencode('.jpeg', frame_)
    image_base64 = base64.b64encode(buffer)
    string = image_base64.decode('utf-8')
    return string


@app.route('/get_zones', methods=['POST'])
def zone_capture():
    global currentConnections
    device_id=request.form['device_id']
    stream_url=request.form['url']
    userid=request.form['userid']
    zones=request.form['activity_area']
    # image1=request.files['image']
    # print('here',zones)
    zone_id=request.form['zone_id']
    action=request.form['action']
    # if(str(userid) in currentConnections):
    #     currentConnections[str(userid)][str(stream_url)]["zones"] = eval(zones)
    
    if(str(userid) in currentConnections):
        if action =='delete':
            currentConnections[str(userid)][str(stream_url)]["zones_dict"].pop(zone_id)
        else:
            currentConnections[str(userid)][str(stream_url)]["zones_dict"][zone_id] = eval(zones)
            print(currentConnections[str(userid)][str(stream_url)]["zones_dict"][zone_id])
   
        

    # keys_to_drop = ['frames', 'previous_frame']
    # # new_currentConnections = currentConnections.copy()
    # new_currentConnections={}
    # for user_data in new_currentConnections.values():
    #     for stream_data in user_data.values():
    #         # if 'counter' in stream_data:
    #         #     del stream_data['counter']
    #         for key in keys_to_drop:
    #             stream_data.pop(key)

    zones_only_dict = {}

    for user_id, streams in currentConnections.items():
        user_zones = {}
        for stream_url, info in streams.items():
            if 'zones_dict' in info:
                user_zones[stream_url] = {'zones_dict': info['zones_dict']}
        zones_only_dict[user_id] = user_zones
    print(zones_only_dict)            
    file_path = 'user_info.json'
    with open(file_path, 'w') as json_file:
        json.dump(zones_only_dict, json_file)
    # with open(file_path, 'w') as file:
    #     for key, value in zones_only_dict.items():
    #         file.write(f"{key}: {value}\n")

    
    return zones
@app.route('/get_notification', methods=['POST'])
def get_notification():
    global currentConnections
    device_id=request.form['device_id']
    stream_url=request.form['url']
    userid=request.form['userid']
    notification_list=request.form['notifications']

    # if(str(userid) in currentConnections):
    #     currentConnections[str(userid)][str(stream_url)]["notifications"] = notification_list

    return notification_list


@app.route('/start_stream', methods=['POST'])
def hello_world():
    global currentConnections
    stream_url=request.form['stream_url']
    userid=request.form['userid']
    try:
        device_n=request.form['device_name']
    except:
        device_n=None
    # away_mode=request.form['away_mode']#for intruder
    # stream_url="rtmp://18.221.116.244/live/main-cam-38"
    # stream_url='parcel.mp4'
    # userid=38
    protocol = re.match(r"(.*?):", stream_url).group(1)
    print(currentConnections)
    print('here',stream_url)


    # if not currentConnections[str(userid)][str(stream_url)]["zones_dict"]: 
    


    #     if str(stream_url) in currentConnections[str(userid)]:
    #         pass
    # else:
    #     if userid in data and stream_url in data[userid]:
    #         zones_dict_value = data[userid][stream_url]['zones_dict']
    #         print("Zones Dict:", zones_dict_value)


    t = Thread(target=livestream, args=(stream_url,userid,device_n))
    # t11 = Thread(target=save_cloud_rtps, args=(userid,stream_url))
    t.start()

    
    return 'Hello World'

if __name__ == '__main__':
   app.debug = True
#    app.run(port=8000)
   app.run(host='0.0.0.0' , port=8000)
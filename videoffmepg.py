import subprocess
import glob
import os
import re
# import ffmpeg
# import sys
from pathlib import Path
import cv2
import numpy as np
import shutil
import time


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def create_video(dirpath,framelist,userid,type):
    print('***********************')


    video_count = os.listdir('videos/{0}'.format(str(userid)))
    length = len(video_count)


    path = os.path.abspath("").replace("\\", "/")


    duration = 0.05
    DIR = "images/{0}".format(str(userid))
    dir='images/' + str(userid) + '/' + str(type) + '/'+ str(length)
    if os.path.exists(dir):
        shutil.rmtree(dir)
        time.sleep(1)
    if not os.path.exists(dir):
        os.makedirs(dir)
        time.sleep(1)
        # print('checking 1')
        # if not os.path.exists(dir):
        #     time.sleep(1)
        #     print('sleeping')
    #after we are successful in getting good stream i need to remove this skipping frames thing 
    for i,newframe in enumerate(framelist):
        len_total=len(framelist)
        num=(1/100)*len_total
        print('****** len',len_total)
        print('********num',num)
        if round(i%num)==0:
            # nparr=np.fromstring(newframe,np.uint8)
            # cv2_image=cv2.imdecode(nparr,cv2.IMREAD_ANYCOLOR)


            cv2.imwrite(str(dir)+'/'+"test"+str(i)+".jpeg",newframe)
            # print('checking2')#,str(dir)+'/'+"test"+str(i)+".jpeg")
            # cv2.imwrite(DIR+"/"+type+"/test"+str(i)+".jpeg",cv2_image)########

        

        # i+=1

    filenames = sorted(os.listdir(dir ), key=numericalSort)
    # filenames = sorted(os.listdir(DIR+"/"+type+"/"), key=numericalSort)
    # video_count=os.listdir('videos/{0}'.format(str(userid)))
    # length = len(video_count)

    script_file = "ffmpeg_input"+str(userid)+".txt"

    with open(script_file, "wb") as outfile:
        for filename in filenames:
            # filename=DIR+"/"+type+"/"+str(filename)
            filename=dir+'/'+str(filename)
            # print('checking3')
            outfile.write(f"file '{filename}'\n".encode())
            outfile.write(f"duration {duration}\n".encode())

    #########################################
    files = os.listdir('videos/' + str(userid))
    l = len(files)
    for k in range(l):
        if os.path.exists('images/'+str(userid)+"/"+type+"/"+str(k)):

            # print('checking4 ')
            shutil.rmtree('images/'+str(userid)+"/"+type+"/"+str(k))
    # imgs=os.listdir('images/'+str(userid)+"/"+type+"/"+str(userid)+'_'+str(l)+'/')
    # for f in imgs:
    #     # if(os.path.exists(f)):
    #     os.rmdir('New folder')
    #     os.remove('images/'+str(userid)+"/"+type+"/"+str(userid)+'_'+str(l)+'/'+str(f))
    ##################################################
    # ffmpeg = os.system("../ffmpeg/ffmpeg")
    # print('checking5')
    vfilename = "{0}{1}.mp4".format(type,length)
    command_line = f"../ffmpeg/ffmpeg -r 22 -f concat -safe 0 -i {script_file} -c:v libx265 -pix_fmt yuv420p videos/{userid}/{vfilename}"
    # command_line = f"../ffmpeg/ffmpeg -r 22 -f concat -safe 0 -i {str(dir) + '/'}test%d.jpeg -c:v libx265 -pix_fmt yuv420p videos/{userid}/{vfilename}"
    # command_line = f"../ffmpeg -r 22 -f concat -safe 0 -i {script_file} -c:v libx265 -pix_fmt yuv420p videos/{userid}/{vfilename}"
    pipe = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE).stdout
    pipe.close()
def backup_clip(dirpath,framelist,userid,type):
    print('*********backup**************')


    video_count = os.listdir('videos/{0}'.format(str(userid)))
    length = len(video_count)


    path = os.path.abspath("").replace("\\", "/")


    duration = 0.05
    DIR = "images/{0}".format(str(userid))
    dir='images/' + str(userid) + '/' + str(type) + '/'+ str(length)
    if os.path.exists(dir):
        shutil.rmtree(dir)
        time.sleep(1)
    if not os.path.exists(dir):
        os.makedirs(dir)
        time.sleep(1)
        print('checking 1')
        # if not os.path.exists(dir):
        #     time.sleep(1)
        #     print('sleeping')
    #after we are successful in getting good stream i need to remove this skipping frames thing 
    for i,newframe in enumerate(framelist):
        len_total=len(framelist)
        if len_total>20:
            num=(1/100)*len_total
            print('****** len',len_total)
            print('********num',num)
            if round(i%num)==0:
                # nparr=np.fromstring(newframe,np.uint8)
                # cv2_image=cv2.imdecode(nparr,cv2.IMREAD_ANYCOLOR)


                cv2.imwrite(str(dir)+'/'+"test"+str(i)+".jpeg",newframe)
                # print('checking2')#,str(dir)+'/'+"test"+str(i)+".jpeg")
                # cv2.imwrite(DIR+"/"+type+"/test"+str(i)+".jpeg",cv2_image)########
        else:
            nparr=np.fromstring(newframe,np.uint8)
            cv2_image=cv2.imdecode(nparr,cv2.IMREAD_ANYCOLOR)
            try:
                cv2.imwrite(str(dir)+'/'+"test"+str(i)+".jpeg",cv2_image)
            except:
                break

        

        # i+=1

    filenames = sorted(os.listdir(dir ), key=numericalSort)
    # filenames = sorted(os.listdir(DIR+"/"+type+"/"), key=numericalSort)
    # video_count=os.listdir('videos/{0}'.format(str(userid)))
    # length = len(video_count)

    script_file = "ffmpeg_input"+str(userid)+".txt"

    with open(script_file, "wb") as outfile:
        for filename in filenames:
            # filename=DIR+"/"+type+"/"+str(filename)
            filename=dir+'/'+str(filename)
            # print('checking3')
            outfile.write(f"file '{filename}'\n".encode())
            outfile.write(f"duration {duration}\n".encode())

    #########################################
    files = os.listdir('videos/' + str(userid))
    l = len(files)
    for k in range(l):
        if os.path.exists('images/'+str(userid)+"/"+type+"/"+str(k)):

            # print('checking4 ')
            shutil.rmtree('images/'+str(userid)+"/"+type+"/"+str(k))
        if os.path.exists('backup/'+str(userid)+"/"+str(k)):
            shutil.rmtree('backup/'+str(userid)+"/"+str(k))


    # imgs=os.listdir('images/'+str(userid)+"/"+type+"/"+str(userid)+'_'+str(l)+'/')
    # for f in imgs:
    #     # if(os.path.exists(f)):
    #     os.rmdir('New folder')
    #     os.remove('images/'+str(userid)+"/"+type+"/"+str(userid)+'_'+str(l)+'/'+str(f))
    ##################################################
    # ffmpeg = os.system("../ffmpeg/ffmpeg")
    # print('checking5')
    vfilename = "backup_{0}{1}.mp4".format(type,length)
    command_line = f"../ffmpeg/ffmpeg -r 22 -f concat -safe 0 -i {script_file} -c:v libx265 -pix_fmt yuv420p videos/{userid}/{vfilename}"
    # command_line = f"../ffmpeg/ffmpeg -r 22 -f concat -safe 0 -i {str(dir) + '/'}test%d.jpeg -c:v libx265 -pix_fmt yuv420p videos/{userid}/{vfilename}"
    # command_line = f"../ffmpeg -r 22 -f concat -safe 0 -i {script_file} -c:v libx265 -pix_fmt yuv420p videos/{userid}/{vfilename}"
    pipe = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE).stdout
    # return_code = pipe.wait()

    # if return_code == 0:
    #     print("-------------FFmpeg completed successfully.------------")
    # # print('///////',pipe)
    pipe.close()
    print('-----------this-----------')







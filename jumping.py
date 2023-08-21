import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os 
# import tensorflow as tf
# print(tf.__version__)
# print(tf.keras.__version__)
LRCN_model=tf.keras.models.load_model('LRCN_model___Date_Time_2023_05_29__14_57_56___Loss_0.8405796885490417___Accuracy_0.6000000238418579 (1).h5',compile=False)#,custom_objects={'Adam':optimizer})
CLASSES_LIST = ["jumping","climbing","running"]


def remove_extra_elements(lst):
    
    diff = len(lst) - 20
    lst = lst[:-diff]
    return lst


def create_video(frames, fps=5):#25
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    count=len(os.listdir('videos'))
    output_path = f'videos/jumping_video_{count}.mp4'
    print(output_path)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()




def predict_single_action(frame_list):
    # print('...........')
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    # video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    # original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    # original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # output_path = "output_video.mp4"
    
    # Declare a list to store video frames we will extract.

    frames_list_normalized = []
    
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    # video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    # skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)
    # print('.......')
    # Iterating the number of times equal to the fixed length of sequence.
    if len(frame_list)>20:
        frame_list = remove_extra_elements(frame_list)
    if len(frame_list)==20:
        for frame_counter in range(20):

            # Set the current frame position of the video.
            # video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

            # Read a frame.
            # success, frame = video_reader.read() 

            # Check if frame is not read properly then break the loop.
            # if not success:
            #     print('//////')
            #     break
            frame=frame_list[frame_counter]
            # Resize the Frame to fixed Dimensions.
            resized_frame = cv2.resize(frame, (64, 64))
            
            # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
            normalized_frame = resized_frame / 255
            
            # Appending the pre-processed frame into the frames list
            frames_list_normalized.append(normalized_frame)
            # print('........')
    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list_normalized, axis = 0))[0]
    print(predicted_labels_probabilities)

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)
    # create_video(frame_list)
    if (predicted_label !=2) and (predicted_labels_probabilities[predicted_label]>=0.80):
        print('........')

        # Get the class name using the retrieved index.
        predicted_class_name = CLASSES_LIST[predicted_label]
        
        # Display the predicted action along with the prediction confidence.
        print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        create_video(frame_list)
     
        return True
        
    # Release the VideoCapture object. 
    # video_reader.release()
    # return
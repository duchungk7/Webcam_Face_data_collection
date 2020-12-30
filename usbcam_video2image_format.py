'''
Last modified time: 2019/06/04
'''
import cv2
import os
import time
from cv2_face_detection import *

# open USB webcam
video_width = 640
video_height = 480
cap = cv2.VideoCapture(0)   # device number 0
#cap = cv2.VideoCapture(0)   # device number 0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)
print("capture device is open: " + str(cap.isOpened()))
# print('CV_CAP_PROP_BRIGHTNESS ', cap.get(cv2.CAP_PROP_BRIGHTNESS))    # BRIGHTNESS: 128(Default)
# print('CV_CAP_PROP_FPS ', cap.get(cv2.CAP_PROP_FPS))      #FPS: 30(Default)

# 影片輸出編碼方式 MP4V
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  


def cap_video(video_name):
    '''
    Open usb webcam and write a video.

    input:

        video name: string 
    '''
    # 建立 VideoWriter 物件, FPS 值為 30.0，解析度為 640x480
    out = cv2.VideoWriter('./video/' + video_name + '.mp4', fourcc, 30.0, (video_width, video_height))

    frame_count = 1     # count number of frame
    start_video = False  # start or not
    
    # loop to read image 
    while True:

        # keyboard input value
        key = cv2.waitKey(1) & 0xFF

        # read from camera
        ret, frame = cap.read()
        if ret:  # success read
            if start_video:     # start
                out.write(frame)
                frame_count += 1

            cv2.imshow('frame', frame)
        else:
            print('camera read fail.')
            break

        if frame_count > 450:   # FPS: 30, so 15秒產生 450張
            break
        if key == ord('q'):     # press 'q' to leave while
            break
        if key == ord(' '):     # press 'space' to start
            print('Starting capture video...')
            start_video = True

    # release VideoWriter object clsoe all windows
    out.release()
    cv2.destroyAllWindows()


def video2image(v_name_list):
    '''
    Divide the video frame into images.

    input:

        v name list: a list contains videos full file path
    '''

    for idx in range(0, len(v_name_list)):
        v_name = v_name_list[idx]

        img_out_dir = "./image/" + v_name
        img_out_face_dir = "./image_face/" + v_name
        img_out_32_dir = "./image_32/" + v_name

        if not os.path.isdir(img_out_dir):
            os.mkdir(img_out_dir)
        if not os.path.isdir(img_out_face_dir):
            os.mkdir(img_out_face_dir)
        if not os.path.isdir(img_out_32_dir):
            os.mkdir(img_out_32_dir)

        print()
        print("video name:", v_name + '.mp4')
        video_cap = cv2.VideoCapture('video/' + v_name + '.mp4')
        success, image = video_cap.read()   # read video
        if not success:
            print("read video error.")
        else:
            print("read video success.")
        
        print("Start dividing the video frame into images...")
        frame_count = 0
        while success:
            # save frame as png file
            cv2.imwrite(img_out_dir + "/" + v_name + "_frame_%d.png" % frame_count, image)
            
            # find face
            find_face, image_face = image_format_face(image)
            if find_face:
                # save face image
                cv2.imwrite(img_out_face_dir + "/" + v_name + "_frame_%d.png" % frame_count, image_face)

                image_face_gray = cv2.cvtColor(image_face, cv2.COLOR_BGR2GRAY)   # GRAY
                image_face_gray_32 = cv2.resize(image_face_gray, (32, 32), interpolation=cv2.INTER_CUBIC)  # resize
                # save face gray 32 * 32 image
                cv2.imwrite(img_out_32_dir + "/" + v_name + "_frame_%d.png" % frame_count, image_face_gray_32)

            success, image = video_cap.read()
            frame_count += 1
        print("total frame number: %d." % frame_count)


def image_format_face(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     # GRAY
    find_face, f_loc = face_detection(gray_img)
    image_fc = []
    if find_face:
        f_loc = box_reduce(f_loc, 30)
        image_fc = face_cut(f_loc, image)
    
    return find_face, image_fc

def check_user_input(u_name):
    '''
    Text interaction to check user(file) name

    input:

        u name: string, user name(first is null)
    '''

    u_name = input("Please input user name [{}]: ".format(u_name)) or u_name
    while u_name == "":     # Prevent no input
        u_name = input("Please input user name [{}]: ".format(u_name)) or u_name
    
    r_angle = ""
    while not r_angle.isdigit():    # Prevent no digit
#        r_angle = input("Please input user rotation angle [{}]: ".format('10')) or '10'

#    return r_angle, u_name
    return u_name


def check_output_dir():
    '''
    Check output dir
    '''
    dir_list = ['./video', './image', './image_face', './image_32']

    for p in dir_list:
        if not os.path.isdir(p):
            os.mkdir(p)


if __name__ == '__main__':
    check_output_dir()

    video_name_list = []
    Back_ground = "GN"
    user_name = ''
    
    while True:
        print("Background:", Back_ground)
#        rotation_angle, user_name = check_user_input(user_name)
#        video_name = Back_ground + '_' + rotation_angle + "_" + user_name
        
        user_name = check_user_input(user_name)
        video_name = Back_ground + '_' + user_name
        
        out_time = "D" + time.strftime('%Y%m%d_%H%M%S', time.localtime())
        video_name = out_time + '_' + video_name
        print("video_name:", video_name)
        video_name_list.append(video_name)

        cap_video(video_name)

        cn = input("Do you want to continue? [y]: (y/n) ") or 'y'
        if cn == 'n':
            break

    # release USB web camera
    cap.release()

    # # if read video false, modify video_name_list here
    # video_name_list = ['D20190501_113338_GN_10_Mia',
    #                    'D20190501_113443_GN_10_Mia',
    #                    'D20190501_113509_GN_10_Mia',
    #                    'D20190501_113549_GN_10_Mia']
    # video_name_list = ['D20190604_174455_GN_10_test',
    #                    'D20190604_174649_GN_10_test']
    video2image(video_name_list)

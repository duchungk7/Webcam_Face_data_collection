'''
Last modified time: 2019/04/25
OpenCV face detection, 
function: face_detection, box_reduce, face_cut, face_combine
'''
import cv2
import os
import numpy as np


# Create the cascade
# Please confirm that the cascPath's file exists.
cascPath = "./frontalface_xml/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


def face_detection(gray_image):
    '''
    OpenCV face detection

    input:

        gray image: array, one gray image
        
    output:

        find face: bool, find a face or no.
        face local: list, the Largest area face bounding box: [left, top, right, bottom]
    '''
    # Detect faces in the gray_image
    faces = faceCascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120),
        #minSize=(120, 120),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # print("Found {0} faces!".format(len(faces)))

    find_face = False
    face_local = []
    if len(faces) > 0:
        find_face = True

        max_size = 120 * 120
        for (x, y, w, h) in faces:      # find max area face
            if w * h >= max_size:
                face_local = list((x, y, x + w, y + h))

    return find_face, face_local


def box_reduce(f_lc, cut_percent=30):
    '''
    Let face bounding box more close to the face, 
    cut percent 30% while reduce the area by 30% 

    input:

        f lc: face bounding box: [left, top, right, bottom]
        cut percent: reduce percent(Default: 30%)

    output:

        f lc: list, face bounding box: [left, top, right, bottom]
    '''

    box_x_len = f_lc[2] - f_lc[0]
    box_y_len = f_lc[3] - f_lc[1]
    cut_x_len = int(box_x_len * (cut_percent / 100))
    cut_y_len = int(box_y_len * (cut_percent / 100))
    f_lc[0] = f_lc[0] + int(cut_x_len / 2)
    f_lc[1] = f_lc[1] + int(cut_y_len / 2)
    f_lc[2] = f_lc[2] - int(cut_x_len / 2)
    f_lc[3] = f_lc[3] - int(cut_y_len / 2)

    return f_lc


def face_cut(f_lc, img):
    '''
    input:

        f lc: face bounding box: [left, top, right, bottom]
        img: source image (BGR or Gray)
    
    output:

        image face: a face image (BGR, Gray match source image)
    '''

    image_face = img[f_lc[1] : f_lc[3], f_lc[0] : f_lc[2]]
    
    return image_face


def face_combine(f_lc, img, back):
    '''
    Let face in background's center

    input:

        f lc: face bounding box: [left, top, right, bottom]
        img: source image (BGR or Gray)
        back: background image (BGR, Gray match source image)
    
    output:

        combine bk: a background image with face (BGR, Gray match background image)
    '''

    box_x_len = f_lc[2] - f_lc[0]
    box_y_len = f_lc[3] - f_lc[1]

    back_shape = np.shape(back)
    start_x = int((back_shape[0] - box_x_len) / 2)
    start_y = int((back_shape[1] - box_y_len) / 2)

    image_face = img[f_lc[1] : f_lc[3], f_lc[0] : f_lc[2]]
    
    combine_bk = back.copy()
    combine_bk[start_y: start_y + box_y_len,
               start_x: start_x + box_x_len] = image_face

    return combine_bk


if __name__ == "__main__":

    ### read a folder all image ###
    # list folder all image
    imagePath_full_list = []
    imagePath_list = os.listdir('./data/box_cut_test')
    for obj in imagePath_list:
        obj = os.path.join('./data/box_cut_test', obj)
        imagePath_full_list.append(obj)

    # background image
    imageBackground = './data/D20190425_GN_Background_480.png'
    image_BK = cv2.imread(imageBackground, 3)

    for p in imagePath_full_list:
        # Read the image
        image = cv2.imread(p, 3)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        success, local = face_detection(gray)
        if success:     # find a face in image
            local = box_reduce(local, 30)
            # img_face = face_cut(local, image)
            # img_cb = face_combine(local, image, image_BK)
            # cv2.imshow(p, img_face)

            cv2.rectangle(image, (local[0], local[1]), (local[2], local[3]), (0, 255, 0), 4, cv2.LINE_AA)

            print(p, 'w:', local[2] - local[0], 'h:', local[3] - local[1])
        else:
            print("No face in picture.")
        cv2.imshow(p, image)

    ### read one image ###
    # imagePath = './test_Wen_frame_0.png'
    # print(imagePath)
    # image = cv2.imread(imagePath, 3)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # success, local = face_detection(gray)
    # if success:   # find a face in image
    #     cv2.rectangle(image, (local[0], local[1]), (local[2], local[3]), (0, 255, 0), 4, cv2.LINE_AA)
    # else:
    #     print("No face in picture.")
    # cv2.imshow("image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

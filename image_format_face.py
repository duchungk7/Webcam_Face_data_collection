'''
Last modified time: 2019/06/04
'''
import os
import cv2
from cv2_face_detection import *


# --- global variable --- #
input_path = "./image"
output_path = "./image_face"


def image_format_face(in_file_dir, out_file_dir):
    '''
    source image(BGR) to face image(BGR)
    '''
    label_list = os.listdir(in_file_dir)   # Test_Max, Test_Wen, Test_Yao
    for index in label_list:
        sub_dir = os.path.join(in_file_dir, index)     # data/test/Test_Max
        img_list = os.listdir(sub_dir)
        print(sub_dir)
        if not os.path.isdir(out_file_dir):
            os.mkdir(out_file_dir)
        output_dir = os.path.join(out_file_dir, index)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for img in img_list:
            # print(img)
            image_data = cv2.imread(os.path.join(sub_dir, img), 3)      # RGB
            gray_img = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)     # GRAY

            find_face, f_loc = face_detection(gray_img)
            if find_face:
                f_loc = box_reduce(f_loc, 100)
                image_fc = face_cut(f_loc, image_data)
                
                cv2.imwrite(os.path.join(output_dir, img), image_fc)
            else:
                print("no face in image.", img)


if __name__ == '__main__':
    image_format_face(input_path, output_path)

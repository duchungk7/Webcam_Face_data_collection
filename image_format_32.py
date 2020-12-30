'''
Last modified time: 2019/06/04
'''
import os
import cv2


# --- global variable --- #
input_path = "./image_face"
output_path = "./image_32"

resize_dim = 128


def image_format_128(in_file_dir, out_file_dir):
    '''
    source image(BGR) to 128x128 image(Gray)
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
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)   # GRAY

            resize_image = cv2.resize(image_data, (resize_dim, resize_dim), interpolation=cv2.INTER_CUBIC)     # resize

            cv2.imwrite(os.path.join(output_dir, img), resize_image)


if __name__ == '__main__':
    image_format_32(input_path, output_path)

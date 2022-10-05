import os, glob
import cv2, argparse
from typing import List
from multiprocessing import Pool
import time

def convert_bb(dh, dw, bbox) -> List[float]:
    """
    draws bounding boxes on the image and returns the image with bounding box on it
    :param image: numpy array of the image
    :param bboxes: List[List[float]] -- list of bounding boxes
    :return: numpy array of image with bounding boxes drawn on it
    """
    
    x, y, w, h = bbox
    l = (x - w / 2) * dw
    r = (x + w / 2) * dw
    t = (y - h / 2) * dh
    b = (y + h / 2) * dh
    
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1
    
    return [int(l), int(t), int(r), int(b)]


def bb(image_path: str, label_path: str) -> List[float]:
    """
    extracts details of all bounding boxes from the label path and returns its list
    :param image_path: str, is the path of image
    :param label_path: str, is the path of label (annotation)
    :return: List[List[float]] which is the list of all bounding boxes,
    where, each bounding box is a list of [x, y, w, h] in the normalised form as is in the yolo format
    """
    img = cv2.imread(image_path)
    dh, dw, _ = img.shape

    # reads the label.txt
    fl = open(label_path, 'r')
    data = fl.readlines()
    fl.close()

    lines = ''
    linesep = ' '

    for dt in data:
        line = []
        ctype, x, y, w, h = map(float, dt.split(' '))
        line.append(str(1+int(ctype)))
        xmin, ymin, xmax, ymax = list(map(str, convert_bb(dh, dw, [x,y,w,h])))
        for _ in [xmin, ymin, xmax, ymax]:
            line.append(_)        
        lines +=  linesep + ' '.join(line)
    lines += '\n'
    return lines


def mywritefunc(imagepath, labelpath, filename):
    # read every label, find every image, write on the ans.txt
    imgbasename = os.path.basename(imagepath)+ ',' 
    with open(filename, 'a') as f:
        if not os.path.exists(labelpath):            
            f.write(imgbasename + '\n')
        else:
            lines = bb(imagepath, labelpath)
            f.write(imgbasename + lines[1:])    
    f.close()


def func(I, L, filename):
    if os.path.exists(filename):
        print('file already there. Deleting!')
        os.remove(filename)

    label_ext = '.txt'
    for image_path in glob.glob(os.path.join(I, '*.*g')):
        image_basename = os.path.basename(image_path)
        label_basename = os.path.splitext(image_basename)[0] + label_ext
        label_path = os.path.join(L, label_basename)
        try:
            mywritefunc(image_path, label_path, filename)
        except Exception:
            print(image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_source', type=str, help='enter image source */images')
    parser.add_argument('--label_source', type=str, help='enter label source */labels')
    opt = parser.parse_args()
    
    SUBMISSION_DIR = f'/usr/src/app/yolov7/runs/submissions'

    ans_path = os.path.join(SUBMISSION_DIR, opt.label_source.split(os.path.sep)[-2]+'.txt')
    if os.path.exists(ans_path):
        print('File already exists. So deleting!')
        os.remove(ans_path)

    print(f'Started writing in {ans_path} .')
    start_time = time.perf_counter()
    func(opt.image_source, opt.label_source, ans_path)
    end_time = time.perf_counter()

    print(f'Time taken {round(end_time-start_time, 2)} sec.\nResults saved in {ans_path} .')

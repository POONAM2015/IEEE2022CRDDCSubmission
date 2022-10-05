# %%
import albumentations as A
import argparse, random
import os, shutil, cv2, glob
import time
from typing import List
from multiprocessing.pool import Pool
from collections import defaultdict
import matplotlib.pyplot as plt

def clamp(x):
    return min(max(0.0, x), 1.0)

# %%
# write augmented images and annotations
def write_augmented_images_bboxes(i, old_image_name, image_ext, timage, bboxes, parent_folder, IMGS, LABELS, AUGMENTED_DATASET):
    # print(f"\tInside write_aug")
    image_basename = 'aug_' + old_image_name + '_' + str(i) + '.' + image_ext
    label_basename = 'aug_' + old_image_name + '_' + str(i) + '.txt'
    aug_image_path = os.path.join(parent_folder, AUGMENTED_DATASET, IMGS, image_basename)
    aug_label_path = os.path.join(parent_folder, AUGMENTED_DATASET, LABELS, label_basename)

    # print(aug_image_path)
    # print(aug_label_path)
    cv2.imwrite(aug_image_path, timage)

    f = open(aug_label_path, 'w')
    with open(aug_label_path, 'w') as f:
        lines = []
        for bbox in bboxes:
            x, y, w, h, cat_idx = map(str, bbox)
            l = [cat_idx, x, y, w, h]
            line = ' '.join(l) + '\n'
            lines.append(line)
        f.writelines(lines)

# %%
def show_image_annotation(image, bboxes):
    # change palette based on number of classes
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    dh, dw, _ = image.shape
    
    for dt in bboxes:
        x, y, w, h, cat_idx = dt
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        cv2.rectangle(image, (l, t), (r, b), palette[cat_idx], 2)
    return image

# %%
def show(image, bboxes, t_image, t_bboxes):
    show_image_annotation(image, bboxes)
    show_image_annotation(t_image, t_bboxes)

    plt.figure(figsize=(16,16))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title('Original image')

    plt.subplot(1,2,2)
    plt.imshow(t_image)
    plt.title('Augmented image')

    plt.show()

# %%
def bounding_box_class_labels(image_path, label_path) -> List[float]:
    img = cv2.imread(image_path)

    fl = open(label_path, 'r')
    data = fl.readlines()
    fl.close()

    bounding_boxes = []

    for dt in data:
        cat_idx, x, y, w, h = map(float, dt.split(' '))
        x, y, w, h = map(clamp, [x, y, w, h])
        bounding_boxes.append([x,y,w,h, int(cat_idx)])    
    return bounding_boxes

# %%
# augmentation super list
aug_super_list = [    
    A.augmentations.transforms.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
    A.augmentations.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5), 
    A.augmentations.transforms.Downscale (scale_min=0.30, scale_max=0.80, interpolation=None, always_apply=False, p=0.5),
    A.augmentations.transforms.Emboss (alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.5), 
    A.augmentations.transforms.Equalize (mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.5), 
    A.augmentations.transforms.FancyPCA (alpha=0.1, always_apply=False, p=0.3),
    A.augmentations.transforms.HueSaturationValue (hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
    A.augmentations.transforms.MedianBlur (blur_limit=3, always_apply=False, p=0.5), 
    A.augmentations.transforms.MultiplicativeNoise (multiplier=(0.9, 1.1), per_channel=False, elementwise=False, always_apply=False, p=0.5),
    A.augmentations.transforms.Posterize (num_bits=4, always_apply=False, p=0.5), 
    A.augmentations.transforms.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
    A.augmentations.transforms.RandomContrast (limit=0.2, always_apply=False, p=0.5),
    A.augmentations.transforms.RandomGamma (gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5), 
    A.augmentations.transforms.RandomToneCurve (scale=0.1, always_apply=False, p=0.5), 
    A.augmentations.transforms.RGBShift (r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5), 
    A.augmentations.transforms.UnsharpMask (blur_limit=(3, 7), sigma_limit=0.0, alpha=(0.2, 0.5), threshold=10, always_apply=False, p=0.5),
    A.augmentations.transforms.HorizontalFlip(p=1),
]


# %%
def single_image_augmentation(arg):
    i, image_name, image_ext, image_path, label_path, parent_folder, IMGS, LABELS, AUGMENTED_DATASET, show_flag = arg
    image = cv2.imread(image_path)

    bboxes = bounding_box_class_labels(image_path, label_path)    
   
    # defining an augmentation pipeline
    aug_dist = defaultdict(int)
    n = 3
    aug_pipeline = random.sample(aug_super_list, n)
    # print('-------------------------------------')
    # print(*aug_pipeline, sep='\n')
    for a in aug_pipeline:
        aug_dist[a] += 1
    try:
        transform = A.Compose(aug_pipeline, bbox_params=A.BboxParams(format='yolo', min_area=10, min_visibility=0.10))
        transformed = transform(image=image, bboxes=bboxes)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        # print(f'NAME - {image_name}/{i}, IMAGE- {image.shape}, BBOXES = {bboxes}\nTRANSFORMED IMAGE- {transformed_image.shape}, T_BBOXES = {transformed_bboxes}\n')
        write_augmented_images_bboxes(i, image_name, image_ext, transformed_image, transformed_bboxes, parent_folder, IMGS, LABELS, AUGMENTED_DATASET)
        
        if show_flag:
            show(image, bboxes, transformed_image, transformed_bboxes)
        return aug_dist
    except Exception:
        print('Can not augment this.')

# %%
def data_augmentation(k, img_source, show_flag, seq_flag):

    if img_source.endswith('.txt'):
        suf = os.path.basename(img_source[:-4])
        with open(img_source, 'r') as f:
            imagepaths = f.readlines()
        imagepaths = [imagepath[:-1] for imagepath in imagepaths]
        parent_folder = os.path.join(img_source, '..')
    else:
        suf = ''
        imagepaths = glob.glob(os.path.join(img_source, '*.*g*'))
        parent_folder = img_source[:-7]
    
    parent_folder = os.path.normpath(parent_folder)
    print(f'Parent folder: {parent_folder}')
    print(f'len of images: {len(imagepaths)}')
    # print(*imagepaths, sep='\n')

    IMGS = 'images'
    LABELS = 'labels'
    AUGMENTED_DATASET = f'AugData_{suf}'

    # make augmented folder
    aug_folder = os.path.join(parent_folder, AUGMENTED_DATASET)
    aug_folder = os.path.normpath(aug_folder)
    if os.path.exists(aug_folder):
        shutil.rmtree(aug_folder)
    print(f'Augmented folder: {aug_folder}')
    os.mkdir(aug_folder)
    for subdir in [IMGS, LABELS]:
        d = os.path.join(aug_folder, subdir)
        os.mkdir(d)
    
    arg_list = []
    for image_path in imagepaths:
        # print(image_path)
        image_basename = os.path.basename(image_path)
        basename, image_ext = image_basename.split('.')
        label_basename = basename + '.txt'
        img_parent_folder = '/'.join(image_path.split('/')[:-2])
        label_path = os.path.join(img_parent_folder, LABELS, label_basename)
        # print(label_path)
        # print('------------')

        if not os.path.exists(label_path):
            print('labelpath no there')
            continue
        
        for i in range(k):
            arg = (i+1, basename, image_ext, image_path, label_path, parent_folder, IMGS, LABELS, AUGMENTED_DATASET, show_flag)
            arg_list.append(arg)

    print(f'No of original images = {len(imagepaths)}\nNo of augmented images to be made = {len(arg_list)}')
    
    if seq_flag:
        print(f'Doing sequentially.')
        list(map(single_image_augmentation, arg_list))
    else:
        print(f'Using multithreading.')
        with Pool() as pool:
            print(f'Pool loop {pool}')
            start_time_ = time.perf_counter()
            list(pool.imap_unordered(single_image_augmentation, arg_list))
            print(f'It took {time.perf_counter() - start_time_:.2} after making the pool')
    # except Exception:
    #     print(f"ERROR found in augmenting -- {image_basename}")

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_source', type=str, default='rdd22/Dataset/Augmented_dataset/JP_US_D20_D40.txt',help='folder/*.txt containing images list')
    parser.add_argument('--label_source', type=str, default='', help='None if images, labels are in the same folder, folder/*.txt containing images list')
    parser.add_argument('--n_aug', type=int, default=2, help='max number of augmented images to obtain from an image')
    parser.add_argument('--seq', action='store_true', help='to perform sequentially')
    opt = parser.parse_args()

    show_flag = False # to show augmented images while being created

    start_time = time.perf_counter()
    data_augmentation(opt.n_aug, opt.image_source, show_flag, opt.seq)
    end_time = time.perf_counter()
    print(f'Finished augmentation in {round(end_time - start_time, 2)} sec.')

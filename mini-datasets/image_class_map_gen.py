import scipy.misc as m
import numpy as np
import os


def get_classes(file_name):
    f_in = open(file_name, 'r')
    return map(int, f_in.readline().split(","))


val_classes = get_classes('val_classes.txt')
train_classes = get_classes('train_classes.txt')


val_path = '/Users/pc/Downloads/ADEChallengeData2016/annotations/validation/'
val_images = os.listdir(val_path)
train_path = '/Users/pc/Downloads/ADEChallengeData2016/annotations/training/'
train_images = os.listdir(train_path)

with open('val_class_map.csv', 'w') as f_out:
    for image_file in val_images:
        img_id = image_file[:-4].split('_')[-1]
        img = m.imread(val_path + image_file)
        img = np.array(img, dtype=np.uint8)
        origin_classes = list(np.unique(img))
        filter_classes = [x for x in origin_classes if x not in train_classes]
        if len(filter_classes) == 1:
            continue
        classes = ",".join([str(x) for x in filter_classes])
        f_out.write(img_id + '-' + classes + '\n')

print val_classes

with open('train_class_map.csv', 'w') as f_out:
    for image_file in train_images:
        img_id = image_file[:-4].split('_')[-1]
        img = m.imread(train_path + image_file)
        img = np.array(img, dtype=np.uint8)
        origin_classes = list(np.unique(img))
        filter_classes = [x for x in origin_classes if x not in val_classes]
        if len(filter_classes) == 1:
            continue
        classes = ",".join([str(x) for x in filter_classes])
        f_out.write(img_id + '-' + classes + '\n')

print train_classes



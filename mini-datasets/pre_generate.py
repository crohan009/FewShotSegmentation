import random

def get_id_and_classes(line):
    # Input: str in the format "<img_id>-class1,class2,...,classk"
    # Returns: (<img_id>, [class1,class2,...,classk])
    lst = line.rstrip().split('-')
    return lst[0], map(int, lst[1].split(','))


def read_classes(class_file):
    with open(class_file,'r') as f_in:
        return map(int, f_in.readline().split(","))


def read_map(map_file):
    with open(map_file, 'r') as f_in:
        class_map = []
        lst = f_in.readlines()
        for i in lst:
            img, classes = get_id_and_classes(i)
            class_map.append([img, classes])
        return class_map


def select_test(classes, test_img_pool):
    random.shuffle(test_img_pool)
    for i in test_img_pool:
        if set(classes).issubset(i[1]):
            return i[0]


def select_train(classes, train_img_pool, img_num):
    result = []
    random.shuffle(train_img_pool)
    for i in train_img_pool:
        if len(set(classes).intersection(i[1])) > 3:
            result.append(i[0])
            img_num -= 1
            if img_num == 0:
                return result


def generate_pre(class_num, img_num, pre_size, classes_pool, test_img_pool, train_img_pool, write_file):
    with open(write_file, 'w') as f_out:
        for i in range(pre_size):
            random.shuffle(classes_pool)
            classes = random.sample(classes_pool, class_num)
            test = select_test(classes, test_img_pool)
            if test:
                print test
            train = select_train(classes, train_img_pool, img_num)
            if test and train:
                print train
                f_out.write(test + ',' + ",".join([x for x in train]) + '\n')


val_test_img = read_map('val_test_map.txt')
val_train_img = read_map('val_train_map.txt')
val_classes = read_classes('val_classes.txt')
val_pre_file = 'val_presentations.txt'
val_presentation_number = 200000000
class_num = 5
img_per_pre = 5

generate_pre(class_num, img_per_pre, val_presentation_number, val_classes, val_test_img, val_train_img, val_pre_file)


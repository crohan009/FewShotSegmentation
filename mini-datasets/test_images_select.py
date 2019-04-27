
def get_id_and_classes(line):
    # Input: str in the format "<img_id>-class1,class2,...,classk"
    # Returns: (<img_id>, [class1,class2,...,classk])
    lst = line.rstrip().split('-')
    return lst[0], map(int, lst[1].split(','))


def get_subset(lists):
    subset_map = []
    for i in lists:
        subsets = []
        for j in lists:
            if i[0] != j[0] and set(j[1]).issubset(i[1]):
                # print i[0], j[0]
                subsets.append(j[0])
        subset_map.append([i[0], subsets])
    subset_map.sort(key=lambda t: len(t[1]), reverse=True)
    return subset_map


def read_classes(file):
    lst = []
    with open(file, 'r') as f_in:
        maps = f_in.readlines()
        for i in maps:
            img, classes = get_id_and_classes(i)
            lst.append([img, classes])
    return lst


def write_subset_map(subset_map, file_name):
    with open(file_name, 'w') as f_out_map:
        for i in subset_map:
            if len(i[1]) < 5:
                continue
            subsets = ",".join([str(x) for x in i[1]])
            f_out_map.write(i[0] + '-' + subsets + '\n')


def gen_class_map(img_list_file, class_map_file, sub_map_file):
    with open(img_list_file, 'r') as f_img_list, \
         open(class_map_file, 'r') as f_class_map, \
         open(sub_map_file, 'w') as f_sub_map:

        img_list = f_img_list.readline().split(',')
        print len(img_list)
        class_map = []
        for i in f_class_map.readlines():
            class_map.append(get_id_and_classes(i))
        for i in img_list:
            for j in class_map:
                if i == j[0]:
                    f_sub_map.writelines(i + '-' + ",".join([str(x) for x in j[1]]) + '\n')


def separate_test(test_size, subset_map, test_file, train_file):
    with open(test_file, 'w') as f_out_test, open(train_file, 'w') as f_out_train:
        test = ",".join([i[0] for i in subset_map[:test_size]])
        f_out_test.writelines(test)
        train = ",".join([i[0] for i in subset_map[test_size:]])
        f_out_train.writelines(train)


val_test_size = 320
train_test_size = 3340


val_classes = read_classes('val_class_map.csv')  # return a list: [image_number, [class0, class1 ...]]
train_classes = read_classes('train_class_map.csv')  # class0, class1... : classes the image contains


val_subset_map = get_subset(val_classes)  # return a list: [image_number, [image0, image1 ...]]
train_subset_map = get_subset(train_classes)  # image0, image1... : images whose classes are subset of the main image


# write_subset_map(val_subset_map, 'val_subset_map.txt')  # write subset map to file
# write_subset_map(train_subset_map, 'train_subset_map.txt')

separate_test(val_test_size, val_subset_map, 'val_test_images.txt', 'val_train_images.txt')
separate_test(train_test_size, train_subset_map, 'train_test_images.txt', 'train_train_images.txt')
# write test images and train images to files separately

# gen_class_map('val_test_images.txt', 'val_class_map.csv', 'val_test_map.txt')
# gen_class_map('val_train_images.txt', 'val_class_map.csv', 'val_train_map.txt')
# gen_class_map('train_test_images.txt', 'train_class_map.csv', 'train_test_map.txt')
# gen_class_map('train_train_images.txt', 'train_class_map.csv', 'train_train_map.txt')


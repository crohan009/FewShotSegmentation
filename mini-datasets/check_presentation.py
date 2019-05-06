

def get_images(line):
    return line.rstrip().split(',')


with open('train_presentations.txt', 'r') as f_in, open('train_pres.txt', 'w') as f_out, \
     open('train_class_list.txt', 'r') as f_class_in, open('train_cl_lst.txt', 'w') as f_class_out:
    old = f_in.readlines()
    old_class = f_class_in.readlines()
    # print old
    lst_old = []
    lst_new = []
    class_old = []
    class_new = []
    for i in old:
        # print i
        lst_old.append(get_images(i))
    for i in old_class:
        class_old.append(get_images(i))

    for i, k in zip(lst_old, class_old):
        flag = False
        for j in lst_new:
            if set(j) == set(i):
                flag = True
                break
        if not flag:
            lst_new.append(i)
            f_out.write(",".join([x for x in i]) + '\n')
            f_class_out.write(",".join([x for x in k]) + '\n')

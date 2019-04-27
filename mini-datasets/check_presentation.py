

def get_images(line):
    return line.rstrip().split(',')


with open('val_presentations.txt', 'r') as f_in, open('val_pres.txt', 'w') as f_out:
    old = f_in.readlines()
    # print old
    lst_old = []
    lst_new = []
    for i in old:
        # print i
        lst_old.append(get_images(i))
    for i in lst_old:
        flag = False
        for j in lst_new:
            if set(j) == set(i):
                flag = True
                break
        if not flag:
            lst_new.append(i)
            f_out.write(",".join([x for x in i]) + '\n')
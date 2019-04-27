import random

li = range(1, 151)
random.shuffle(li)

val = li[0:50]
val.sort()
train = li[50:150]
train.sort()

f_out_val = open('val_classes.txt', 'w')
val_classes = ",".join([str(i) for i in val])
f_out_val.writelines(val_classes)
f_out_val.close()

f_out_train = open('train_classes.txt', 'w')
train_classes = ",".join([str(i) for i in train])
f_out_train.writelines(train_classes)
f_out_train.close()

print len(val), len(train)
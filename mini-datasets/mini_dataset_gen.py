# import scipy.misc as m
# import numpy as np
# import os
from random import shuffle


def get_id_and_classes(line):
	# Input: str in the format "<img_id>-class1,class2,...,classk"
	# Returns: (<img_id>, [class1,class2,...,classk])
	lst = line.rstrip().split('-')
	return lst[0], lst[1].split(',')

def gen_topN_unique_classes(f_name, num_classes=5):
	lst = []
	with open(f_name, 'r') as fd:
		fd.readline()
		for i in range(num_classes):
			line = fd.readline()
			lst.append(line.split('\t')[0])
	#lst.sort()
	return lst

def gen_file_list(f_name):
	lst = []
	with open(f_name, 'r') as fd:
		for line in fd:
			lst.append(line.rstrip())
	return lst

def a_subset_b(a, b):
	# returns True is a is a subset of b
	for i in a:
		if i not in b:
			return False
	return True


def gen_single_presentation(img_id, img_classes, shuffled_file_lst, pres_size=6, class_per_pres=6):
	pres_images = [img_id]
	for line in shuffled_file_lst:
		curr_img, curr_img_classes = get_id_and_classes(line)
		if (curr_img not in pres_images) and a_subset_b(curr_img_classes, img_classes):
			pres_images.append(curr_img)
			if len(pres_images) == pres_size:
				break
	return pres_images


def generate_dataset(in_file, out_file, class_lst, pres_size=6, class_per_pres=6):
	file_lst_shuffle = gen_file_list(in_file)   # shuffled repeatedly
	file_lst_ordered = gen_file_list(in_file)	# left alone

	total_classes = len(class_lst)
	#num_req_presentations = int(len(file_lst_ordered) / class_per_pres)
	print("Generating {}".format(out_file))
	f_out = open(out_file, 'w')
	#print(len(total_classes))
	#print("num_req_presentations = {}".format(num_req_presentations))

	pres_ctr = 0
	for line in file_lst_ordered:
		img, img_classes = get_id_and_classes(line)
		#print(img_classes, class_lst)
		#break
		if len(img_classes) <= class_per_pres and a_subset_b(img_classes, class_lst) :
			#print(img)
			shuffle(file_lst_shuffle)
			pres_lst = gen_single_presentation(img, img_classes, file_lst_shuffle, pres_size, class_per_pres)
			if len(pres_lst) == pres_size:
				print(','.join(pres_lst), file=f_out)
				pres_ctr +=1

	print("\t {} presentations created.".format(pres_ctr))
	f_out.close()

def avg_classes(file):
	file_lst = gen_file_list(file)
	summ = 0
	for line in file_lst:
		img, img_classes = get_id_and_classes(line)
		summ += len(img_classes)
	return int(summ / len(file_lst))


if __name__ == "__main__":
	file1 = "indices_of_ADEChallengeData2016_sortedByVal.txt"
	file2 = "image_class_map.csv"

	class_nums = [21, 51, 101, 151]

	topN_class_lst = ['0'] + gen_topN_unique_classes(file1, 150)


	print("Top 20 class occurances = ", topN_class_lst[:21])
	avg_class_per_img = avg_classes(file2)
	print("Average classes per image = {}".format(avg_class_per_img))

	nbshots = [1, 3, 5, 10] # Number of 'shots' in the few-shots learning

	for i in range(len(class_nums)):
		generate_dataset(in_file = file2, 
			out_file= "dataset{}_top{}.txt".format(i+1, class_nums[i]-1),
			class_lst= topN_class_lst[:class_nums[i]],
			pres_size= 6, 
			class_per_pres= 6
			)

    # data_path = '/scratch/rc3232/advml/project/pytorch-semseg/ptsemseg/datasets/ADEChallengeData2016/annotations/validation/'
    # images = os.listdir(data_path)
    
    # with open('image_class_map.csv', 'w') as fd:
    #     for image_file in images:
    #         img_id = image_file[:-4].split('_')[-1]
    #         img = m.imread(data_path + image_file)
    #         img = np.array(img, dtype=np.uint8)
    #         classes = ",".join([str(i) for i in list(np.unique(img))])
    #         print("{}-{}".format(img_id, classes))
            
            
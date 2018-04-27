import random
import numpy as np

server = False

if server:
    path = '/'
else:
    path = '/mnt/UCF-11/'
    out_file = 'data/listfile.txt'
    train_file = 'data/trainlist.txt'
    test_file = 'data/testlist.txt'
    class_file = 'data/classInd.txt'

classInd=[]

with open(class_file) as f0:
    for line in f0:
        # class_name = line.split(' ')[1]
        class_name = line.rstrip()
        if class_name:
            classInd.append(class_name)

x=0
y=0
z=0

dis = []

for i in range(11):
    dis.append([])

with open(out_file) as f1:
    for line in f1:
    	if line == '':
    		continue
        folder = line.split('/')[1]
    	class_name = line.split('/')[0].rstrip()
        class_id = classInd.index(class_name)
    	arr_folder = folder.split('_')
    	num = int(arr_folder[len(arr_folder) - 1])
    	
    	dis[class_id].append([line.rstrip() + ' ' + str(class_id) + '\n', num])

ind_group = []
for i in range(11)
    temp = np.arange(1,26)
    random.shuffle(temp)
    ind_group.append(temp)

while True:
    sum1 = 0
    sum2 = 0
    test = []
    train = []

    file_train = open(train_file,'w')
    file_test = open(test_file,'w')

    for i in range(11):
        length = len(dis[i])
        random.shuffle(ind_group)
        pior = np.random.randint(17,19)

        for j in range(length):
            if (dis[i][j][1] in ind_group[:pior]):
                file_train.write(dis[i][j][0])
                sum1 += 1
            else:
                file_test.write(dis[i][j][0])
                sum2 += 1

    print(sum1, sum2)
    file_train.close()
    file_test.close()
    if sum1 == 1120:
        break



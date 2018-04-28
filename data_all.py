import numpy as np 
import pickle
import sys
import config

if len(sys.argv) < 3:
    print 'Missing agrument'
    print 'Ex: data_all.py train run'
    sys.exit()

train = sys.argv[1]
debug = sys.argv[2]
server = config.server()
output_path = config.data_output_path()


opt_file = r'{}database/{}-opt4.pickle'.format(output_path,train)
out_file = r'{}database/{}-all.pickle'.format(output_path,train)

with open(opt_file,'rb') as f1:
    opt = pickle.load(f1)

l = len(opt)
opt_all = []
for i in range(l):
		x = int(np.floor(opt[i][1]*1.0/5))*5
		opt_all.append([
			opt[i][0], (x*2), opt[i][2], opt[i][3], 
			(x*4), (x*2), opt[i][1]
		])
		opt_all.append([
			opt[i][0], (x*2 + 10), opt[i][2], opt[i][3],
			(x * 4 + 20), (x*2 + 10), opt[i][1]
		])

print len(opt_all)
if debug == 'run':
	with open(out_file,'wb') as f3:
	    pickle.dump(opt_all,f3)
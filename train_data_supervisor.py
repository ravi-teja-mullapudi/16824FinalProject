import numpy as np

# Make sure that caffe is on the python path:
import sys
import caffe

data_listfile = '/home/rmullapu/16824FinalProject/superclass_labels_train.txt'

caffe.set_device(0)
caffe.set_mode_gpu()
net_names = []
net_list = []

net_list.append(caffe.Net('/home/rmullapu/16824FinalProject/models/compute_increase/2_unit_deploy.prototxt',
                 '/home/rmullapu/16824FinalProject/2_unit/_iter_30000.caffemodel',
                 caffe.TEST))
net_names.append('2_unit')

net_list.append(caffe.Net('/home/rmullapu/16824FinalProject/models/compute_increase/1_unit_deploy.prototxt',
                 '/home/rmullapu/16824FinalProject/1_unit/_iter_40000.caffemodel',
                 caffe.TEST))
net_names.append('1_unit')

net_list.append(caffe.Net('/home/rmullapu/16824FinalProject/models/bvlc_reference_caffenet/deploy.prototxt',
                 '/home/rmullapu/16824FinalProject/alex_net/_iter_50000.caffemodel',
                 caffe.TEST))
net_names.append('alex_net')

net_list.append(caffe.Net('/home/rmullapu/16824FinalProject/models/vgg/vgg_deploy.prototxt',
                 '/home/rmullapu/16824FinalProject/vgg/_iter_20000.caffemodel',
                 caffe.TEST))
net_names.append('vgg')

data_list = np.loadtxt(data_listfile,  str, comments=None, delimiter='\n')
data_counts = len(data_list)
for net in net_list:
    assert(net_list[0].blobs['data'].data.shape[0] == 64)

batch_size = net_list[0].blobs['data'].data.shape[0]
batch_count = int(np.ceil(data_counts * 1.0 / batch_size))

print 'num_batches: ' + str(batch_count)

f_list = []
for n in xrange(0, len(net_list)):
    f_list.append(open('%s_out_label.txt'%(net_names[n]), 'w'))

scores = {}
total = {}
min_net = {}
for n in xrange(0, len(net_list)):
    scores[net_names[n]] = {}
    total[net_names[n]] = {}

for i in range(batch_count):
    print 'batch_num ' + str(i)
    for n in xrange(0, len(net_list)):
        out = net_list[n].forward()
        for j in range(batch_size):
            id = i * batch_size + j
            if id >= data_counts:
                break

            lbl = int(data_list[id].split(' ')[1])
            fname = data_list[id].split(' ')[0]
            prop = out['softmax'][j]
            pred_lbl = prop.argmax()
            f_list[n].write('%s %d\n'%(fname, pred_lbl))
            if pred_lbl == lbl:
                if lbl in scores[net_names[n]]:
                    scores[net_names[n]][lbl] = scores[net_names[n]][lbl] + 1
                else:
                    scores[net_names[n]][lbl] = 1

                if fname in min_net:
                    min_net[fname] = min(min_net[fname], n)
                else:
                    min_net[fname] = n

            if lbl in total[net_names[n]]:
                total[net_names[n]][lbl] = total[net_names[n]][lbl] + 1
            else:
                total[net_names[n]][lbl] = 1

for n in xrange(0, len(net_list)):
    f_list[n].close()

f = open('meta_labels.txt', 'w')
for name in min_net:
    f.write('%s %d\n'%(name, min_net[name]))
f.close()

f = open('network_acc.txt', 'w')
for net_name, val in scores.iteritems():
    f.write('%s\n'%(net_name))
    for c in val:
        f.write('%s %f\n'%(c, float(val[c])/total[net_name][c]))
f.close()

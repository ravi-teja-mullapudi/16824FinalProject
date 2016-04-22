from __future__ import division
import sys

caffe_root = '/home/rmullapu/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

# init
caffe.set_mode_gpu()
caffe.set_device(0)

# caffe.set_mode_cpu()

solver = caffe.SGDSolver('/home/rmullapu/16824FinalProject/models/12class/12class_solver.prototxt')

for layer_name, blob in solver.net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

niter = 50000
train_loss = np.zeros(niter)

f = open('12class_log.txt', 'w')

for it in range(niter):
    solver.step(1)
    train_loss[it] = solver.net.blobs['loss'].data
    f.write('{0: f}\n'.format(train_loss[it]))

f.close()
# solver.step(80000)



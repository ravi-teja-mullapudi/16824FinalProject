import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt',
                'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

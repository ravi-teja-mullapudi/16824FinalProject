import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net('models/vgg_deploy.prototxt',
                'models/VGG_ILSVRC_16_layers.caffemodel',
                caffe.TEST)

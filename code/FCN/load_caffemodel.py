# The function was found in https://stackoverflow.com/questions/45199643/how-do-i-load-a-caffe-model-and-convert-to-a-numpy-array

from caffenet import *
import caffe
import numpy as np


def shai_net_to_py_readable(prototxt_filename, caffemodel_filename):
    net = caffe.Net(prototxt_filename, caffemodel_filename, caffe.TEST) # read the net + weights
    pynet_ = []
    for li in xrange(len(net.layers)):  # for each layer in the net
        layer = {}    # store layer's information
        layer['name'] = net._layer_names[li]
        # for each input to the layer (aka "bottom") store its name and shape
        layer['bottoms'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape)
                             for bi in list(net._bottom_ids(li))]
        # for each output of the layer (aka "top") store its name and shape
        layer['tops'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape)
                          for bi in list(net._top_ids(li))]
        layer['type'] = net.layers[li].type  # type of the layer
        # the internal parameters of the layer. not all layers has weights.
        layer['weights'] = [net.layers[li].blobs[bi].data[...]
                            for bi in xrange(len(net.layers[li].blobs))]
        pynet_.append(layer)
    return pynet_


pynet = shai_net_to_py_readable('deploy_FCN.prototxt', 'FCN_iter_160000.caffemodel')
print(pynet)
np.save('FCN_weights.npy', pynet)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, logging
import tensorflow as tf
import tensorflow.contrib.slim as slim


def git_root():
    ''' return the root location of git rep
    '''
    import subprocess
    gitroot = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
    return gitroot



def _depthwise_separable_conv(
    inputs,
    num_pwc_filters,
    width_multiplier,
    sc,
    downsample=False):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = 2 if downsample else 1

    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=_stride,
                                                  depth_multiplier=1,
                                                  kernel_size=[3, 3],
                                                  scope=sc+'/depthwise_conv')

    bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
    pointwise_conv = slim.convolution2d(bn,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        scope=sc+'/pointwise_conv')
    bn = slim.batch_norm(pointwise_conv, scope=sc+'/Regularizer')
    return bn



def _initalize_variables(hypes):
    if hypes['load_pretrained']:
        print("Pretrained weights are loaded.")
        print("The model is fine-tuned from previous training.")
        restore = hypes['restore']
        init = tf.global_variables_initializer()
        sess = tf.get_default_session()
        sess.run(init)

        saver = tf.train.Saver(var_list=restore)
        logging.info("Restored list:{}".format(restore))
        #filename = git_root()+"/checkpoints/MobileNetCheckpoint/mobilenet_v1_1.0_224.ckpt"
        #logging.info("Loading weights from disk.")
        #saver.restore(sess, filename)
    else:
        logging.info("Random initialization performed.")
        sess = tf.get_default_session()
        init = tf.global_variables_initializer()
        sess.run(init)



def inference(
    hypes,
    images,
    train=True,
    num_classes=1000,
    num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
    preprocess=True,
    bottleneck=True):


    net = mobilenet(
        images,
        is_training=train,
        num_classes=num_classes)

    if train:
        hypes['init_function'] = _initalize_variables
        hypes['restore'] = tf.global_variables()

    dct = dict()
    dct['deep_feat'] = net.avg_pool_15
    dct['early_feat'] = net.conv_ds_14
    dct['net'] = net
    return dct



def get_image_as_tensor(image_paths):
    filename_queue = tf.train.string_input_producer(image_paths) #  list of files to read
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        res = []
        for i in range(image_paths): #length of your filename list
            res.append(my_img.eval()) #here is your image Tensor :)

    coord.request_stop()
    coord.join(threads)
    return res

class mobilenet(object):
    def predict(self, image_path):
        images = get_image_as_tensor([image_path])


    def __init__(self, inputs,
        is_training=True,
        num_classes=1000,
        width_multiplier=1,
        scope='MobileNet'):

        self.inputs = inputs
        with tf.variable_scope(scope) as sc:
            end_points_collection = sc.name + '_end_points'

        with slim.arg_scope(
            [slim.convolution2d, slim.separable_convolution2d],
            activation_fn=None,
            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.batch_norm],
                is_training=is_training,
                activation_fn=tf.nn.relu,
                fused=True):
                self.conv_1 = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME', scope='conv_1')
                self.conv_1_batch_norm = slim.batch_norm(self.conv_1, scope='conv_1/batch_norm')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.conv_1_batch_norm) #This registers it to norm dict
                self.conv_ds_2 = _depthwise_separable_conv(self.conv_1_batch_norm, 64, width_multiplier, sc='conv_ds_2')
                self.conv_ds_3 = _depthwise_separable_conv(self.conv_ds_2, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                self.conv_ds_4 = _depthwise_separable_conv(self.conv_ds_3, 128, width_multiplier, sc='conv_ds_4')
                self.conv_ds_5 = _depthwise_separable_conv(self.conv_ds_4, 256, width_multiplier, downsample=True, sc='conv_ds_5')
                self.conv_ds_6 = _depthwise_separable_conv(self.conv_ds_4, 256, width_multiplier, sc='conv_ds_6')
                self.conv_ds_7 = _depthwise_separable_conv(self.conv_ds_6, 512, width_multiplier, downsample=True, sc='conv_ds_7')

                self.conv_ds_8 = _depthwise_separable_conv(self.conv_ds_7, 512, width_multiplier, sc='conv_ds_8')
                self.conv_ds_9 = _depthwise_separable_conv(self.conv_ds_8, 512, width_multiplier, sc='conv_ds_9')
                self.conv_ds_10 = _depthwise_separable_conv(self.conv_ds_9, 512, width_multiplier, sc='conv_ds_10')
                self.conv_ds_11 = _depthwise_separable_conv(self.conv_ds_10, 512, width_multiplier, sc='conv_ds_11')
                self.conv_ds_12 = _depthwise_separable_conv(self.conv_ds_11, 512, width_multiplier, sc='conv_ds_12')

                self.conv_ds_13 = _depthwise_separable_conv(self.conv_ds_12, 1024, width_multiplier, downsample=True, sc='conv_ds_13')
                self.conv_ds_14 = _depthwise_separable_conv(self.conv_ds_13, 1024, width_multiplier, sc='conv_ds_14')
                self.avg_pool_15 = slim.avg_pool2d(self.conv_ds_14, [3, 3], scope='avg_pool_15', padding='SAME')

        self.end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        #self.SpatialSqueeze = tf.squeeze(self.avg_pool_15, [1, 2], name='SpatialSqueeze')
        self.logits = slim.fully_connected(self.avg_pool_15, num_classes, activation_fn=None, scope='fc_16')
        self.predictions = slim.softmax(self.logits, scope='Predictions')




def mobilenet_arg_scope(weight_decay=0.0):
  """Defines the default mobilenet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the MobileNet model.
  """
  with slim.arg_scope(
      [slim.convolution2d, slim.separable_convolution2d],
      weights_initializer=slim.initializers.xavier_initializer(),
      biases_initializer=slim.init_ops.zeros_initializer(),
      weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
    return sc


if __name__=='__main__':
    import json, os, sys
    from scipy import misc
    hypes = json.load(open('../hypes/kittiBox_Mobilenet.json','rt'));

    '''filenames = ['/image_dir/img.jpg']
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    images = tf.image.decode_png(value, channels=3)
    '''
    h, w = 224, 224
    h, w = hypes['image_height'], hypes['image_width']

    img = misc.imread('../data/demo.png')
    img = misc.imresize(img , [h, w])
    x = tf.placeholder(tf.float32, [None, h , w, 3])
    dct = inference(hypes = hypes,
        images = x,
        train=False)
    self = dct['net']

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(self.predictions, feed_dict={
            self.inputs:img.reshape([1, h , w, 3])
            })
        #print (res)


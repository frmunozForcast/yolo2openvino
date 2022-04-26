import tensorflow as tf
from models.common import _conv2d_fixed_padding, _fixed_padding, _get_size, \
    _detection_layer, _upsample

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

def _tiny_res_block(inputs,in_channels,data_format):

    net = _conv2d_fixed_padding(inputs,in_channels,kernel_size=3)
    route = net
    #_,split=tf.split(net,num_or_size_splits=2,axis=1 if data_format =="NCHW" else 3)
    split = net[:, in_channels//2:, :, :]if data_format=="NCHW" else net[:, :, :, in_channels//2:]
    net = _conv2d_fixed_padding(split,in_channels//2,kernel_size=3)
    route1 = net
    net = _conv2d_fixed_padding(net,in_channels//2,kernel_size=3)
    net = tf.concat([net, route1], axis=1 if data_format == 'NCHW' else 3)
    net = _conv2d_fixed_padding(net,in_channels,kernel_size=1)
    feat = net
    net = tf.concat([route, net], axis=1 if data_format == 'NCHW' else 3)
    net = slim.max_pool2d(
        net, [2, 2], scope='pool2')
    return net, feat


def yolo_v4_tiny(inputs, num_classes, anchors, masks, is_training=False, data_format='NCHW', reuse=False, **kwargs):
    """
    Creates YOLO v4 model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param num_classes: anchors.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :param with_spp: whether or not is using spp layer.
    :param masks: list specifying masks in the form [[3,4,5], [0,1,2]]
    :param kwargs: additional kwargs for custom yolo v4
    :return:
    """
    # if True, define the net with 3 yolo layers, following the custom model included in darknet
    # https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny-3l.cfg
    three_yolo = kwargs.get("three_yolo", False)
    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding, slim.max_pool2d], data_format=data_format):
        with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
                with tf.variable_scope('yolo-v4-tiny'):
                    #CSPDARKENT BEGIN
                    net = _conv2d_fixed_padding(inputs, 32, kernel_size=3, strides=2)  # layer 1

                    net = _conv2d_fixed_padding(net, 64, kernel_size=3, strides=2)  # layer 2

                    net, _ = _tiny_res_block(net, 64, data_format)  # layers 3-10
                    net, feat3 = _tiny_res_block(net, 128, data_format)  # layers 11-18, feat3 = 15
                    net, feat = _tiny_res_block(net, 256, data_format)  # layers 19-26, feat = 23
                    net = _conv2d_fixed_padding(net, 512, kernel_size=3)  # layer 27
                    feat2 = net
                    #CSPDARKNET END

                    # yolo layer 1
                    net = _conv2d_fixed_padding(feat2, 256, kernel_size=1)
                    route = net
                    net = _conv2d_fixed_padding(route, 512, kernel_size=3)
                    detect_1 = _detection_layer(
                        net, num_classes, anchors[masks[0][0]:masks[0][-1] + 1], img_size, data_format)
                    detect_1 = tf.identity(detect_1, name='detect_1')

                    # yolo layer 2
                    net = _conv2d_fixed_padding(route, 128, kernel_size=1)
                    upsample_size = feat.get_shape().as_list()
                    net = _upsample(net, upsample_size, data_format)
                    net = tf.concat([net, feat], axis=1 if data_format == 'NCHW' else 3)
                    net = _conv2d_fixed_padding(net, 256, kernel_size=3)
                    route2 = net
                    detect_2 = _detection_layer(
                        net, num_classes, anchors[masks[1][0]:masks[1][-1] + 1], img_size, data_format)
                    detect_2 = tf.identity(detect_2, name='detect_2')

                    if three_yolo:
                        # yolo layer 3
                        net = _conv2d_fixed_padding(route2, 64, kernel_size=1)
                        upsample_size = feat3.get_shape().as_list()
                        net = _upsample(net, upsample_size, data_format)
                        net = tf.concat([net, feat3], axis=1 if data_format == 'NCHW' else 3)
                        net = _conv2d_fixed_padding(net, 128, kernel_size=3)
                        detect_3 = _detection_layer(
                            net, num_classes, anchors[masks[2][0]:masks[2][-1] + 1], img_size, data_format)
                        detect_3 = tf.identity(detect_3, name='detect_3')
                        detections = tf.concat([detect_1, detect_2, detect_3], axis=1)

                    else:
                        detections = tf.concat([detect_1, detect_2], axis=1)
                    detections = tf.identity(detections, name='detections')
                    return detections
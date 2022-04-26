import tensorflow as tf
from models import yolo_v3, yolo_v3_tiny, yolo_v4, yolo_v4_tiny

from utils.utils import load_weights, load_names, detections_boxes, freeze_graph
from utils.anchors import Anchors
from utils.masks import Masks
import json

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
    'yolo', 3, 'Yolo version. Possible options: 3 or 4.'
)
tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names. Usually a file ending in .names.')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with Yolo\' weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NHWC', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'output', 'frozen_darknet_yolov3_model.pb', 'Frozen tensorflow protobuf model output path')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of Yolo')
# tf.app.flags.DEFINE_bool(
#    'spp', False, 'Use SPP version of Yolo')
tf.app.flags.DEFINE_integer(
    'size', None, 'Image size. If both, height and width, are not provided, a square input shape of size as set with'
                 ' this flag will be used')
tf.app.flags.DEFINE_integer(
    'height', None, 'Input image height. If height is set, width must also be set. The size flag will be ignored.',
    short_name='h'
)
tf.app.flags.DEFINE_integer(
    'width', None, 'Input image width. If width is set, height must also be set. The size flag will be ignored.',
    short_name='w'
)
tf.app.flags.DEFINE_list(
    'anchors', None, 'List of anchors. If not set default anchors for YoloV3, YoloV4, and YoloV3/V4-tiny will be set.',
    short_name='a'
)
tf.app.flags.DEFINE_string(
    'config', None, 'config file used for defining the network. It can replace the FLAGS inputs.'
)


def main(argv=None):

    if FLAGS.config is not None:
        config = json.load(open(FLAGS.config, "r"))
    else:
        config = {}
    yolo_version = config.pop("yolo", FLAGS.yolo)
    is_tiny = config.pop("tiny", FLAGS.tiny)
    if yolo_version == 3:
        if is_tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
            default_anchors = Anchors.YOLOV3TINY.value
            masks = config.pop("masks", Masks.YOLOV3TINY.value)
        else:
            model = yolo_v3.yolo_v3
            default_anchors = Anchors.YOLOV3.value
            masks = config.pop("masks", Masks.YOLOV3.value)
    elif yolo_version == 4:
        if is_tiny:
            model = yolo_v4_tiny.yolo_v4_tiny
            default_anchors = Anchors.YOLOV4TINY.value
            masks = config.pop("masks", Masks.YOLOV4TINY.value)
        else:
            model = yolo_v4.yolo_v4
            default_anchors = Anchors.YOLOV4.value
            masks = config.pop("masks", Masks.YOLOV4.value)
    else:
        raise ValueError(f"{yolo_version} is not supported Yolo version. Supported versions: 3, 4.")

    input_anchors = config.pop("anchors", FLAGS.anchors)
    selected_anchors = default_anchors if input_anchors is None else [int(a) for a in input_anchors]
    anchors = [(selected_anchors[i * 2], selected_anchors[i * 2 + 1]) for i in range(len(selected_anchors) // 2)]

    num_classes = config.pop("classes", len(load_names(FLAGS.class_names)))
    height = config.pop("height", FLAGS.height)
    width = config.pop("width", FLAGS.width)
    size = config.pop("size", FLAGS.size)

    # set input shape
    if height is not None and width is not None:
        inputs = tf.placeholder(tf.float32, [None, height, width, 3], "inputs")
        if size is not None:
            print("Width and height are set, size flag will be ignored!")
    elif size is not None:
        inputs = tf.placeholder(tf.float32, [None, size, size, 3], "inputs")
    else:
        raise ValueError("width/height are not set. Please specify input shape on config file!")

    with tf.variable_scope('detector'):
        # masks and **config only works on YoloV4 models, on YoloV3 it does nothing
        detections = model(inputs, num_classes, anchors, masks, data_format=FLAGS.data_format, **config)
        load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

    # Sets the output nodes in the current session
    boxes = detections_boxes(detections)

    print("Starting conversion with the following parameters:")
    print(f"Yolo version: {yolo_version}")
    print(f"Tiny: {is_tiny}")
    print(f"Anchors: {anchors}")
    print(f"Masks: {masks}")
    if is_tiny:
        print(f"3 yolo layers: {three_yolo}")
    else:
        print(f"small_objects: {small_objects}")
    print(f"Shape: {inputs}")
    print(f"Classes: {num_classes}")

    with tf.Session() as sess:
        sess.run(load_ops)
        freeze_graph(sess, FLAGS.output)

if __name__ == '__main__':
    tf.app.run()

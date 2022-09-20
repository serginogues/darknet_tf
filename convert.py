"""
https://github.com/hunglc007/tensorflow-yolov4-tflite
"""
import tensorflow as tf
import argparse
from yolo_tf.core.yolov4 import YOLO, decode, filter_boxes
from yolo_tf.core import utils


def run_convert(input: str,
                output: str = 'new_model_tf',
                input_size: int = 416,
                is_tiny: bool = False,
                model: str = 'yolov4'):
    """
    Parameters
    ----------
    input
        path to darknet model
    output
        path to converted tf model
    input_size
        size of YOLO input to resize
    is_tiny
        if True then loads tiny arch
    model
        'yolov3 or yolov4'
    """
    framework = 'tf'
    score_thres = 0.2

    # load config
    NUM_CLASS = len(utils.read_class_names(utils.get_file_path(input, '.names')))
    STRIDES, ANCHORS, XYSCALE = utils.load_config(is_tiny, model)

    # create YOLO architecture
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLO(input_layer, NUM_CLASS, model, is_tiny)

    bbox_tensors = []
    prob_tensors = []
    if is_tiny:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
            else:
                output_tensors = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
            elif i == 1:
                output_tensors = decode(fm, input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
            else:
                output_tensors = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)

    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=score_thres,
                                    input_shape=tf.constant([input_size, input_size]))
    pred = tf.concat([boxes, pred_conf], axis=-1)
    model = tf.keras.Model(input_layer, pred)
    utils.load_weights(model, utils.get_file_path(input, '.weights'), model, is_tiny)
    model.summary()
    model.save(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path to yolo darknet folder including .weights, .names and .cfg",
                        required=True)
    parser.add_argument("--output", type=str, default="checkpoints/new_model-tf", help="path to converted model",
                        required=False)
    parser.add_argument("--input_size", type=int, default=416, help="model input size",
                        required=False)
    parser.add_argument("--is_tiny", type=bool, default=False, required=False)
    parser.add_argument("--model", type=str, default='yolov4', required=False, help="yolov3 or yolov4")

    args = parser.parse_args()
    run_convert(args.input, args.output, args.input_size, args.is_tiny, args.model)
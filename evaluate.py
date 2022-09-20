import argparse
from yolo_tf.generate_predictions import run_generate_predictions
from yolo_tf.map import run_map
from yolo_tf.confusion_matrix import run_confusion_matrix
from yolo_tf.core import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid", type=str, help="path to validation set", required=False)
    parser.add_argument("--model", type=str, help="path to tf model", required=True)
    parser.add_argument("--ignore", type=list[str], default=[], required=False, help="list of class id to ignore")
    parser.add_argument("--gen_pred", type=bool, default=False,
                        help="if True, generates predictions at results/")

    args = parser.parse_args()
    if args.gen_pred:
        run_generate_predictions(args.valid, args.model)
    classes_path = utils.get_file_path(args.model, '.names')
    run_map(classes_path, args.ignore, True, 0.5)
    run_confusion_matrix(classes_path)
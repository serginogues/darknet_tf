import shutil
import glob
import cv2
import os
from tqdm import tqdm
from .network import YOLOwrapper

img_extension = ['.jpg', '.JPG', '.PNG', '.png', '.jpeg']


def run_generate_predictions(validate: str, model: str):
    """
    Write predictions and ground_truth in txt files
    Parameters
    ----------
    validate
        path to valid set
    model
        path to tf model
    """
    print("Generate Predictions")
    print("Building YOLO from: " + model)
    net = YOLOwrapper(model_path=model)

    predicted_dir_path = 'mAP/predicted'
    ground_truth_dir_path = 'mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)

    for img_id, f in enumerate(tqdm(glob.glob(os.path.join(validate, '*[!.txt]')), desc='Creating labels at mAP/')):

        img_path, suffix = os.path.splitext(f)
        if suffix in img_extension:
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            pred_bbox = net.forward(img)  # (n, [y1, x1, y2, x2, score, class_id])

            predict_result_path = os.path.join(predicted_dir_path, str(img_id) + '.txt')
            with open(predict_result_path, 'w') as file:
                for bb in pred_bbox:
                    ymin, xmin, ymax, xmax, score, class_id = bb
                    bbox_mess = ' '.join([str(int(class_id)), '%.4f' % score, str(int(xmin)), str(int(ymin)), str(int(xmax)), str(int(ymax))]) + '\n'
                    file.write(bbox_mess)

            gt_bbox = labelImg_to_yolo_format(read_bboxes(img_path + '.txt'), height, width)
            gt_result_path = os.path.join(ground_truth_dir_path, str(img_id) + '.txt')
            with open(gt_result_path, 'w') as file:
                for bb in gt_bbox:
                    ymin, xmin, ymax, xmax, class_id = bb
                    bbox_mess = ' '.join([str(class_id), str(xmin), str(ymin), str(xmax), str(ymax)]) + '\n'
                    file.write(bbox_mess)


def read_bboxes(path):
    """
    LabelImg objects are stored as (label_id, x, y, width, height). \n
    Returns list of obj as (x, y, width, height, label_id).
    """
    object_list = []
    with open(path, 'r') as data:
        for line in data:
            x = line.split(" ")
            object_list.append([float(x[1]), float(x[2]), float(x[3]), float(x[4]), int(x[0])])
    return object_list


def labelImg_to_yolo_format(bboxes, image_height, image_width):
    """
    (x, y, width, height, label_id) to [ymin, xmin, ymax, xmax, label_id]

    :param bboxes: frame detections
    :param image_height: height of the frame
    :param image_width: width of the frame
    """
    new_boxes = []
    for box in bboxes:
        xcenter = int(box[0] * image_width)
        ycenter = int(box[1] * image_height)
        width = int(box[2] * image_width)
        height = int(box[3] * image_height)

        xmin = xcenter - (width/2)
        ymin = ycenter - (height/2)
        xmax = xcenter + (width/2)
        ymax = ycenter + (height/2)

        new_boxes.append([ymin, xmin, ymax, xmax, box[4]])
    return new_boxes

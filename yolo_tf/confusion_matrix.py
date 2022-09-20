"""
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)
array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]])
"""
import os
from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sn
from sklearn.metrics import confusion_matrix
import glob
from tqdm import tqdm
from .map import file_lines_to_list
from .core import utils


def run_confusion_matrix(classes: str):
    """
    Parameters
    ----------
    classes
        path to .names
    """
    ground_truth_files_list = glob.glob('yolo_tf/mAP/predicted/*.txt')
    ground_truth_files_list.sort()
    CLASSES = utils.read_class_names(classes)

    y_true = []
    y_pred = []
    for file in tqdm(ground_truth_files_list, desc='GT file'):

        # get file id: 0.txt -> 0, 7.txt -> 7
        file_id = file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))

        # <class_name> <conf> <left> <top> <right> <bottom>
        pred_lines = file_lines_to_list(file)

        # <class_name> <left> <top> <right> <bottom>
        gt_lines = file_lines_to_list(os.path.join('yolo_tf/mAP/ground-truth', file_id + '.txt'))

        # add 'used' boolean to avoid assigning a gt_bbox twice
        gt_lines = [[x, False] for x in gt_lines]

        # find predictions match
        for pred_l in pred_lines:
            pred_l = pred_l.split()
            pred_class = int(pred_l[0])
            pred_bb = [float(pred_l[2]), float(pred_l[3]), float(pred_l[4]), float(pred_l[5])]  # <left> <top> <right> <bottom>

            # check first pred class
            is_match = False
            for i in range(len(gt_lines)):
                gt_split = gt_lines[i][0].split()
                gt_class = int(gt_split[0])

                if not gt_lines[i][1] and gt_class == pred_class:
                    gt_bb = [float(gt_split[1]), float(gt_split[2]), float(gt_split[3]), float(gt_split[4])]  # <left> <top> <right> <bottom>
                    iou_ = _IoU(gt_bb, pred_bb)
                    if iou_ > 0.5:
                        y_pred.append(pred_class)
                        y_true.append(gt_class)
                        # gt_line[1] = True
                        gt_lines[i][1] = True
                        is_match = True
                        break

            # if not TP then check for confusions
            if not is_match:
                for i in range(len(gt_lines)):
                    gt_split = gt_lines[i][0].split()
                    gt_class = int(gt_split[0])

                    if not gt_lines[i][1] and gt_class != pred_class:
                        gt_bb = [float(gt_split[1]), float(gt_split[2]), float(gt_split[3]), float(gt_split[4])]  # <left> <top> <right> <bottom>

                        iou_ = _IoU(gt_bb, pred_bb)
                        if iou_ > 0.5:
                            y_pred.append(pred_class)
                            y_true.append(gt_class)
                            break

    [print('predicted ' + str(CLASSES[x]) + ' but gt is ' + str(CLASSES[y_true[idx]])) for idx, x in enumerate(y_pred)
     if x != y_true[idx]]
    cm = confusion_matrix(y_true, y_pred)
    columns = [CLASSES[x] for x in list(set(y_pred))]
    df_cm = DataFrame(cm, index=columns, columns=columns)
    ax = sn.heatmap(df_cm, cmap='Oranges', annot=True, fmt='g')
    plt.xlabel('Predictions')
    plt.ylabel('Ground-Truth')
    plt.savefig("yolo_tf/mAP/results/confusionMatrix.png", bbox_inches='tight', dpi=100)
    print('See results at yolo_tf/mAP/results/')


def _IoU(bbgt, bb):
    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1
    if iw > 0 and ih > 0:
        # compute overlap (IoU) = area of intersection / area of union
        union_area = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                  + 1) * (
                             bbgt[3] - bbgt[1] + 1) - iw * ih
        iou_ = iw * ih / union_area
        return iou_
    else:
        return 0
import glob
import json
import os
import shutil
from .core import utils
from .mAP.map_utils import *


def run_map(names: str, ignore_class: list[str] = [], draw_plot: bool = False, minoverlap: float = 0.5):
    """
    Parameters
    ----------
    names
        path to .names
    ignore_class
        classes id to ignore during mAP computation
    draw_plot
        if True draws plots at ./results
    minoverlap
        IoU threshold
    """

    tmp_files_path = "yolo_tf/mAP/tmp_files"
    results_files_path = "yolo_tf/mAP/results"
    CLASSES = utils.read_class_names(names)
    MINOVERLAP = minoverlap

    """
    Create "tmp_files/" and "results/" directories
    """
    if not os.path.exists(tmp_files_path):  # if it doesn't exist already
        os.makedirs(tmp_files_path)

    if os.path.exists(results_files_path):  # if it exist already
        # reset the results directory
        shutil.rmtree(results_files_path)

    os.makedirs(results_files_path)
    if draw_plot:
        os.makedirs(results_files_path + "/classes")

    """
    Ground-Truth
     Load each of the ground-truth files into a temporary ".json" file.
     Create a list of all the class names present in the ground-truth (gt_classes).
    """
    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob('yolo_tf/mAP/ground-truth/*.txt')
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    for txt_file in ground_truth_files_list:

        # get file id: 0.txt -> 0, 7.txt -> 7
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))

        # check if there is a correspondent predicted objects file
        """if not os.path.exists('predicted/' + file_id + ".txt"):
            error_msg = "Error. File not found: predicted/" + file_id + ".txt\n"
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
            error(error_msg)"""

        # read gt_file
        lines_list = file_lines_to_list(txt_file)

        # create ground-truth dictionary
        bounding_boxes = []
        for line in lines_list:
            # <class_name> <left> <top> <right> <bottom>
            class_name, left, top, right, bottom = line.split()

            # check if class is in the ignore list, if yes skip
            if class_name in ignore_class:
                continue
            bbox = left + " " + top + " " + right + " " + bottom
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})

            # count that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1
        # dump bounding_boxes into a ".json" file
        with open(tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    """
    Predicted
     Load each of the predicted files into a temporary ".json" file.
    """
    predicted_files_list = glob.glob('yolo_tf/mAP/predicted/*.txt')
    predicted_files_list.sort()
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in predicted_files_list:
            # print(txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            lines = file_lines_to_list(txt_file)
            for line in lines:
                tmp_class_name, confidence, left, top, right, bottom = line.split()

                if tmp_class_name == class_name:
                    # print("match")
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": '0.5', "file_id": file_id, "bbox": bbox})
                    # print(bounding_boxes)
        # sort predictions by decreasing confidence
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(tmp_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
    Calculate the AP for each class
    """
    sum_AP = 0.0
    ap_dictionary = {}
    # open file to store the results
    with open(results_files_path + "/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0

            """
            Load predictions of that class
            """
            predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
            predictions_data = json.load(open(predictions_file))

            """
            Assign predictions to ground truth objects
            Count TP and FP
            """
            tp = [0] * len(predictions_data)  # creates an array of zeros of size nd
            fp = [0] * len(predictions_data)
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction["file_id"]
                bb = [float(x) for x in prediction["bbox"].split()]

                # assign prediction to ground truth object if any
                #   open ground-truth with that file_id
                gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                IoU = -1
                gt_match = -1
                # load prediction bounding-box

                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            union_area = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                                      + 1) * (
                                                     bbgt[3] - bbgt[1] + 1) - iw * ih
                            iou_ = iw * ih / union_area
                            if iou_ > IoU:
                                IoU = iou_
                                gt_match = obj

                # assign prediction as true positive/don't care/false positive
                # set minimum overlap
                min_overlap = MINOVERLAP
                if IoU >= min_overlap and not bool(gt_match["used"]):
                    # true positive
                    tp[idx] = 1
                    gt_match["used"] = True
                    count_true_positives[class_name] += 1
                    # update the ".json" file
                    with open(gt_file, 'w') as f:
                        f.write(json.dumps(ground_truth_data))

                else:
                    # false positive
                    fp[idx] = 1

            # print(tp)
            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            # print(tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(val) / gt_counter_per_class[class_name]
            # print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            # print(prec)

            ap, mrec, mprec = voc_ap(rec, prec)
            sum_AP += ap
            text = "{0:.2f}%".format(
                ap * 100) + " = " + CLASSES[int(class_name)] + " AP  "  # class_name + " AP = {0:.2f}%".format(ap*100)

            """
            Write to results.txt
            """
            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")
            ap_dictionary[class_name] = ap

            """
            Draw plot
            """
            if draw_plot:
                plt.plot(rec, prec, '-o')
                # add a new penultimate point to the list (mrec[-2], 0.0)
                # since the last line segment (and respective area) do not affect the AP value
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                # set window title
                fig = plt.gcf()  # gcf - get current figure
                fig.canvas.manager.set_window_title('AP ' + class_name)
                # set plot title
                plt.title('class: ' + text)
                # plt.suptitle('This is a somewhat long figure title', fontsize=16)
                # set axis titles
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                # optional - set axes
                axes = plt.gca()  # gca - get current axes
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                # Alternative option -> wait for button to be pressed
                # while not plt.waitforbuttonpress(): pass # wait for key display
                # Alternative option -> normal display
                # plt.show()
                # save the plot
                fig.savefig(results_files_path + "/classes/" + class_name + ".png")
                plt.cla()  # clear axes for next plot

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes
        text = "mAP = {0:.2f}%".format(mAP * 100)
        results_file.write(text + "\n")

    # remove the tmp_files directory
    shutil.rmtree(tmp_files_path)

    """
    Count total of Predictions
    """
    # iterate through all the files
    pred_counter_per_class = {}
    # all_classes_predicted_files = set([])
    for txt_file in predicted_files_list:
        # get lines to list
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            # check if class is in the ignore list, if yes skip
            if class_name in ignore_class:
                continue
            # count that object
            if class_name in pred_counter_per_class:
                pred_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                pred_counter_per_class[class_name] = 1
    # print(pred_counter_per_class)
    pred_classes = list(pred_counter_per_class.keys())

    """
    Plot the total number of occurences of each class in the ground-truth
    """
    if draw_plot:
        window_title = "Ground-Truth Info"
        plot_title = "Ground-Truth\n"
        plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
        x_label = "Number of objects per class"
        output_path = results_files_path + "/Ground-Truth Info.png"
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(
            gt_counter_per_class,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
            CLASSES
        )

    """
    Write number of ground-truth objects per class to results.txt
    """
    with open(results_files_path + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(CLASSES[int(class_name)] + ": " + str(gt_counter_per_class[class_name]) + "\n")

    """
    Finish counting true positives
    """
    for class_name in pred_classes:
        # if class exists in predictions but not in ground-truth then there are no true positives in that class
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0
    # print(count_true_positives)

    """
    Plot the total number of occurences of each class in the "predicted" folder
    """
    if draw_plot:
        window_title = "Predicted Objects Info"
        # Plot title
        plot_title = "Predicted Objects\n"
        plot_title += "(" + str(len(predicted_files_list)) + " files and "
        count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(pred_counter_per_class.values()))
        plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
        # end Plot title
        x_label = "Number of objects per class"
        output_path = results_files_path + "/Predicted Objects Info.png"
        to_show = False
        plot_color = 'forestgreen'
        true_p_bar = count_true_positives
        draw_plot_func(
            pred_counter_per_class,
            len(pred_counter_per_class),
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            true_p_bar,
            CLASSES
        )

    """
    Write number of predicted objects per class to results.txt
    """
    with open(results_files_path + "/results", 'a') as results_file:
        results_file.write("\n# Number of predicted objects per class\n")
        for class_name in sorted(pred_classes):
            n_pred = pred_counter_per_class[class_name]
            text = CLASSES[int(class_name)] + ": " + str(n_pred)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_pred - count_true_positives[class_name]) + ")\n"
            results_file.write(text)

    """
    Draw mAP plot (Show AP's of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mAP * 100)
        x_label = "Average Precision"
        output_path = results_files_path + "/mAP.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            ap_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            "",
            CLASSES
        )
    print('See results at mAP/results/')


def update_dict(d, classes):
    new_dict = {}
    for x in d.items():
        new_dict[classes[int(x[0])]] = x[1]
    return new_dict

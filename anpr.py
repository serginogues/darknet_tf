import os
from detect import from_video, from_img
from yolo_tf.pipelines import ANPR

INPUT = 'data/PoC/UPC/video2_7_h.mp4'
STRIDE = 2


def main():
    chkp_paths = \
        ['checkpoints/yolov4-416-coco-tf',
         'checkpoints/yolov4-416-licence-plate-tf',
         'checkpoints/yolov4-416-character_segmentation-tf'
        ]
    conf_list = [0.8, 0.15, 0.9]

    model = ANPR(chkp_paths, conf_list)

    if os.path.isdir(INPUT):
        for f in os.listdir(INPUT):
            ff = os.path.join(INPUT, f)
            if os.path.isfile(ff):
                videoORimage(ff, model)
    else:
        videoORimage(INPUT, model)


def videoORimage(path: str, model):
    _, extension = os.path.splitext(path)
    isVid = True if extension.lower() in ['.mp4', '.avi'] else False

    if isVid:
        from_video(model, path, SAVE=True, stride=STRIDE)
    else:
        from_img(model, path, SAVE=False)


if __name__ == '__main__':
    main()
import numpy as np
from .network import YOLOwrapper
import cv2
import re


class ANPR():
    """
    Automatic Number Plate Recognition model based on three stage YOLO detections
    """
    def __init__(self, checkpoints_list: list[str], conf_list: list[float] = [0.8, 0.15, 0.9]):
        """
        Parameters
        ----------
        checkpoints_list
            path list to tf models
        conf_list
            list of model confidences
        """
        self.m1 = YOLOwrapper(
                            model_path=checkpoints_list[0],
                            confidence=conf_list[0]
                            )
        self.m2 = YOLOwrapper(
                            model_path=checkpoints_list[1],
                            confidence=conf_list[1]
                            )
        self.m3 = YOLOwrapper(
                            model_path=checkpoints_list[2],
                            confidence=conf_list[2]
                            )
        self.vehicle_classes = [2, 3, 5, 7]
        self.predictions = []
        self.licence_list = []

    def forward(self, frame: np.ndarray):
        """
        Parameters
        ----------
        frame
            image of shape (height, width, 3)

        Returns
        -------
        np.ndarray
            array of shape (num bboxes, 5) where 5 = (y1, x1, y2, x2, label)
        """
        y1 = self.m1.forward(frame)

        output = []
        for bbox in y1:
            if bbox[5] in self.vehicle_classes:
                new_img = self.crop(frame, bbox)
                y2 = self.m2.forward(new_img)
                label = self.m1.classes[bbox[5]]
                if len(y2) > 0:
                    new_img2 = self.crop(new_img, y2[0])
                    y3 = self.m3.forward(new_img2)

                    if len(y3) > 0:
                        licence = ''.join([i[0] for i in sorted([(self.m3.classes[bb[5]], bb[1]) for bb in y3],
                                                                   key=lambda x: x[1])])
                        label += ', ' + licence + ', ' + str(round(np.min(y3[:, 4]), 2))
                        output.append((*bbox[:4], label))

                        if bool(re.search(r'\d\d\d\d[A-Z][A-Z][A-Z]|[A-Z]\d\d\d\d[A-Z][A-Z]', licence)) \
                                and licence not in self.licence_list:
                            self.licence_list.append(licence)

        self.predictions = np.array(output)
        return self.predictions

    @staticmethod
    def crop(img, bbox):
        return img[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3]), :]

    def draw_predictions(self, image):
        image_h, image_w, _ = image.shape
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        color = (255, 180, 130)
        for bbox in self.predictions:
            c1, c2 = (int(float(bbox[1])), int(float(bbox[0]))), (int(float(bbox[3])), int(float(bbox[2])))
            cv2.rectangle(image, c1, c2, color, bbox_thick)

            bbox_mess = bbox[4]
            fontScale = 0.5
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (int(np.float32(c3[0])), int(np.float32(c3[1]))), color, -1)  # filled
            cv2.putText(image, bbox_mess, (c1[0], int(np.float32(c1[1] - 2))), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

        for idx, licence in enumerate(self.licence_list):
            cv2.putText(image, licence, (int(image_w * 0.05), int(image_h * 0.05 * (idx+1))), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, lineType=cv2.LINE_AA)

        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
__config = ConfigProto()
__config.gpu_options.allow_growth = True
__session = InteractiveSession(config=__config)
from .core import utils


class YOLOwrapper:
    def __init__(self, model_path: str,
                 input_size: int = 416,
                 iou: float = 0.45,
                 confidence: float = 0.3,
                 classes_path: str = None):
        """
        If classes_path == None, then .classes must be inside model_path repository
        """
        self.input_size = input_size
        self.path = model_path
        self.iou = iou
        self.conf = confidence
        self.model = tf.keras.models.load_model(self.path, compile=False)
        print('Model loaded from: ' + model_path)
        self.predictions = []
        self.classes = utils.read_class_names(utils.get_file_path(self.path, '.names')) if classes_path is None \
            else utils.read_class_names(classes_path)

    def preprocess(self, img: np.ndarray):
        """
        Returns
        -------
        np.ndarray
            image batch of shape (1, input_size, input_size, 3)
        """
        image = cv2.resize(img, (self.input_size, self.input_size))
        image = image / 255.
        batch = np.asarray([image]).astype(np.float32)
        tensor = tf.constant(batch)
        return tensor

    def forward(self, x):
        """
        Returns
        -------
        np.ndarray
            shape = (num_boxes, 6) where 6 is (y1, x1, y2, x2, score, class_id)
        """
        image_h, image_w, _ = x.shape

        # preprocess
        x = self.preprocess(x)

        # inference
        y = self.model.predict(x)

        # postprocess
        boxes = y[:, :, 0:4]
        pred_conf = y[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou,
            score_threshold=self.conf
        )
        out_boxes, out_scores, out_classes, num_boxes  = \
            boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()

        output = []
        output2 = []
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > len(self.classes): continue
            coor = out_boxes[0][i]

            # relative y1, x1, y2, x2, score, class_id
            output2.append((*coor[:4], out_scores[0][i], int(out_classes[0][i])))

            coor[0] = round(float(coor[0] * image_h), 1)
            coor[2] = round(float(coor[2] * image_h), 1)
            coor[1] = round(float(coor[1] * image_w), 1)
            coor[3] = round(float(coor[3] * image_w), 1)

            # y1, x1, y2, x2, score, class_id
            output.append((*coor, out_scores[0][i], int(out_classes[0][i])))

        self.predictions = np.array(output)
        return self.predictions

    def draw_predictions(self, frame):
        image = utils.draw_bbox(frame, self.predictions, self.classes)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)




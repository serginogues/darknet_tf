# darknet_tf

## ``convert.py``
You can convert a pretrained Darknet model to Tensorflow with:
```bash
python convert.py 
--input ./checkpoints/yolov4-416-coco-darknet # path to model folder
--output ./checkpoints/yolov4-416-coco-tf # optional, path to folder where the converted tf model is saved
--is_tiny IS_TINY # optional, is yolo-tiny or not
--input_size INPUT_SIZE  # optional, YOLO input size
--model yolov4 # optional, yolov3 or yolov4
```

## Usage

```python
import cv2
from yolo_tf.network import YOLOwrapper

path = './checkpoints/yolov4-416-coco-tf'
conf = 0.2
image_path = 'path/to/image.png'

model = YOLOwrapper(
    model_path=path,
    confidence=conf
)

# read image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# inference
pred_bbox = model.forward(img)

# draw predictions
image = model.draw_predictions(img)

# show
cv2.imshow('', image)
cv2.waitKey(0)
```

## ``detect.py``
```bash
python detect.py 
--input ./data/car_hr.JPG # path to image or video
--model ./checkpoints/yolov4-416-coco-tf # path to tensorflow model folder
--save False # optional, if True saves img or video
```

## Evaluate
First generate predictions (which are saved at ``darknet_tf/mAP/``):
```bash
python evaluate.py
--valid .your/dataset/valid # optional, path to validation folder that includes labels stored as .txt files (one file per image)
--model ./checkpoints/yolov4-416-coco-tf # path to tensorflow model folder
--ignore 4, 8 # optional, list of class id to ignore
--gen_pred False # if True, loads tf model and generates predictions
```

## References
Code based on [github.com/hunglc007/tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)


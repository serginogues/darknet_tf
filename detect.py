import os
import time
import cv2
import argparse
from yolo_tf.network import YOLOwrapper


def run_detect(input: str, model: str, save: bool = False):
    """
    Parameters
    ---------
    input
        path to your image or video
    model
        path to checkpoint repo
    save
        saves output img or video
    """
    model = YOLOwrapper(
        model_path=model,
        confidence=0.2
    )
    _, extension = os.path.splitext(input)
    if extension.lower() in ['.mp4', '.avi']:
        from_video(model, input, SAVE=save)
    else:
        from_img(model, input, SAVE=save)


def from_img(model, image_path: str, SAVE: bool = False):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    prev_time = time.time()
    pred_bbox = model.forward(original_image)
    curr_time = time.time()
    print("time: %.2f ms" % (1000 * (curr_time - prev_time)))
    print(pred_bbox)

    image = model.draw_predictions(original_image)
    cv2.imshow('', image)
    cv2.waitKey(0)
    if SAVE: cv2.imwrite('out.jpg', image)


def from_video(model, vid_path: str, SAVE: bool = False, stride: int = 1):
    vid = cv2.VideoCapture(vid_path)

    if SAVE:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('out.mp4', codec, fps, (width, height))

    frame_id = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            raise ValueError("No image! Try with another video format")

        if frame_id % stride == 0:
            prev_time = time.time()
            model.forward(frame)
            curr_time = time.time()
            print('frame: ' + str(frame_id))
            print("time: %.2f ms" % (1000 * (curr_time - prev_time)))

            result = model.draw_predictions(frame)
        else:
            result = model.draw_predictions(frame)

        if SAVE:
            out.write(result)
        else:
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow('', result)
            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_id += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path to image or video",
                        required=True)
    parser.add_argument("--model", type=str, default="checkpoints/new_model-tf", help="path to tf model",
                        required=True)
    parser.add_argument("--save", type=bool, default=False, required=False)

    args = parser.parse_args()
    run_detect(args.input, args.model, args.save)

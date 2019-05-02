import argparse
import collections
import glob
import sys

from PIL import Image

from yolo import YOLO, detect_video


def detect_img(yolo):
    while True:
        img = input("Input image filename:")
        try:
            image = Image.open(img)
        except:
            print("Open Error! Try again!")
            continue

        r_image = yolo.detect_image(image)
        r_image.show()
    yolo.close_session()


class ResultsLogger:
    def __init__(self, args, yolo):
        self.file = None
        self.yolo = yolo

        if args.csv:
            self.file = open(args.csv, 'w')
            self.file.write('file,total_boxes,' + ','.join(yolo.class_names) + '\n')

    def log(self, filename, classes):
        if self.file is None:
            return

        counter = collections.Counter(classes)
        row = [filename, len(classes)] + [counter[i] for i in range(len(self.yolo.class_names))]
        self.file.write(','.join(map(str, row)) + '\n')

    def close(self):
        if self.file:
            self.file.close()


def process_directory(yolo, args):
    files = sorted(glob.glob(args.directory + '/**/*.*', recursive=True))
    res_logger = ResultsLogger(args, yolo)

    for filename in files:
        try:
            image = Image.open(filename)
        except:
            print("Open Error! Try again!", filename)
            continue

        print(filename)
        try:
            boxes, scores, classes = yolo.process_image(image)
        except Exception:
            # some files are broken, e.g.
            # OSError: image file is truncated (nn bytes not processed).
            print(sys.exc_info())
            classes = []

        if len(boxes):
            print('boxes', boxes)
            print('classes', classes)
            print('scores', scores)

        res_logger.log(filename, classes)

    res_logger.close()
    yolo.close_session()


FLAGS = None

if __name__ == "__main__":
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """
    parser.add_argument(
        "--model",
        type=str,
        help="path to model weight file, default " + YOLO.get_defaults("model_path"),
    )

    parser.add_argument(
        "--anchors",
        type=str,
        help="path to anchor definitions, default " + YOLO.get_defaults("anchors_path"),
    )

    parser.add_argument(
        "--classes",
        type=str,
        help="path to class definitions, default " + YOLO.get_defaults("classes_path"),
    )

    parser.add_argument(
        "--gpu_num",
        type=int,
        help="Number of GPU to use, default " + str(YOLO.get_defaults("gpu_num")),
    )

    parser.add_argument(
        "--image",
        default=False,
        action="store_true",
        help="Image detection mode, will ignore all positional arguments",
    )
    """
    Command line positional arguments -- for video detection mode
    """
    parser.add_argument(
        "--input", help="Video input path",
    )

    parser.add_argument(
        "--output", help="[Optional] Video output path", default=None
    )

    parser.add_argument(
        '--directory',
        help='Process whole directory of images'
    )

    parser.add_argument(
        "--csv", help="Output CSV files for images and detections"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        detect_img(YOLO(**vars(FLAGS)))
    elif 'directory' in FLAGS:
        process_directory(YOLO(**vars(FLAGS)), FLAGS)
    elif 'input' in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")

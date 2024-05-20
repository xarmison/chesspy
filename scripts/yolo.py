import sys, os
# Add the path to the yolo module to the system path
sys.path.insert(1, os.getenv('YOLO_PATH', '~/repos/chesspy/yolo_repo'))

from argparse import Namespace
import numpy as np
import cv2 as cv
import darknet

class Yolo:
        
    def __init__(self, args: Namespace, thresh: float = 0.3) -> None:
        """
            Initializes the YOLO detector.

            Args:
                args (Namespace): The command-line arguments containing the paths to the YOLO configuration, weights, and metadata files.
                thresh (float, optional): The confidence threshold for object detection. Defaults to 0.3.
        """

        self.config_file = args.config_path
        self.weights_file = args.weights_path 
        self.meta_file = args.meta_path
        self.thresh = thresh

        self.network, self.class_names, self.class_colors = darknet.load_network(
            self.config_file, self.meta_file, self.weights_file
        )

        self.network_width = darknet.network_width(self.network)
        self.network_height = darknet.network_height(self.network)

    def detect(self, img: np.array) -> list[dict]:
        """
            Detects objects in an image using the YOLO algorithm.

            Args:
                img (np.array): The input image as a NumPy array.

            Returns:
                list[dict]: A list of dictionaries representing the detected objects. Each dictionary contains the following keys:
                    - 'class_name': The name of the detected object class.
                    - 'confidence': The confidence score of the detection.
                    - 'coords': The coordinates of the bounding box around the detected object in the format (x, y, width, height).
        """

        img_height, img_width, _ = img.shape

        scale_x = img_width / self.network_width
        scale_y = img_height / self.network_height

        # Convert the image to the format required by Darknet
        darknet_image = darknet.make_image(
            self.network_width,
            self.network_height,
            3
        )

        img_resized = cv.resize(
            cv.cvtColor(img, cv.COLOR_BGR2RGB),
            (self.network_width, self.network_height),
            interpolation=cv.INTER_LINEAR
        )

        darknet.copy_image_from_bytes(darknet_image, img_resized.tobytes())

        # Perform object detection
        detections = darknet.detect_image(
            self.network,
            self.class_names,
            darknet_image,
            thresh=self.thresh
        )

        darknet.free_image(darknet_image)

        # Scale detections back to original image coords
        detections_processed = [
            {
                'class_name': class_name, 
                'confidence': confidence, 
                'coords': (
                    round(x * scale_x), 
                    round(y * scale_y), 
                    round(w * scale_x), 
                    round(h * scale_y)
                )
            }
            for class_name, confidence, (x, y, w, h) in detections
        ]

        return detections_processed

    def draw_detection_boxes(self, image: np.array, detections: list[dict]) -> np.array:
        """
            Draws bounding boxes and labels on the input image based on the provided detections.

            Args:
                image (np.array): The input image on which to draw the bounding boxes and labels.
                detections (list[dict]): A list of dictionaries representing the detected objects. Each dictionary should contain the following keys:
                    - 'coords' (tuple): The coordinates of the bounding box in the format (x, y, width, height).
                    - 'class_name' (str): The class name of the detected object.
                    - 'confidence' (float): The confidence score of the detection.

            Returns:
                np.array: The image with the bounding boxes and labels drawn.

        """
        
        fen_piece_map = {
            'white_pawn': 'P',
            'white_rook': 'R',
            'white_knight': 'N',
            'white_bishop': 'B',
            'white_queen': 'Q',
            'white_king': 'K',
            'black_pawn': 'p',
            'black_rook': 'r',
            'black_knight': 'n',
            'black_bishop': 'b',
            'black_queen': 'q',
            'black_king': 'k'
        }
        
        line_tickness = int(min(image.shape[:2]) * 0.003)

        for detection in detections:
            x, y, w, h = detection['coords']
            top_left = (
                int(round(x - (w / 2))),
                int(round(y - (h / 2)))
            )
            bottom_right = (
                int(round(x + (w / 2))),
                int(round(y + (h / 2)))
            )

            cv.rectangle(
                image,
                top_left, bottom_right,
                self.class_colors[detection['class_name']],
                line_tickness
            )

            cv.putText(
                image, 
                f"{fen_piece_map[detection['class_name']]} ({detection['confidence']})",
                (top_left[0], top_left[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                self.class_colors[detection['class_name']], line_tickness
            )

        return image

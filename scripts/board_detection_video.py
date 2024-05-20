from chess import Board, svg, engine, parse_square
from argparse import Namespace, ArgumentParser
from json import load as load_json
from cairosvg import svg2png
from io import BytesIO
from PIL import Image
import numpy as np
import cv2 as cv

from helper_functions import get_lines_intersections, get_squares_coords_dict, \
    detections_to_fen, get_fen_piece_name

from board_selector import roi_selector
from yolo import Yolo


def parse_args() -> Namespace:
    """
        Parses command line arguments for the board detection script.

        Returns:
            argparse.Namespace: Parsed command line arguments.
    """

    parser = ArgumentParser(
        description='Identifies a chess position in each frame of a game recording.'
    )

    parser.add_argument(
        'board_video', type=str,
        help='Path to the video file to be processed.'
    )

    parser.add_argument(
        '--config-path', type=str,
        default='~/repos/chesspy/yolo/configs/yolov4_tiny_chess.cfg',
        help='Path to the yolo configuration file.'
    )

    parser.add_argument(
        '--weights-path', type=str,
        default='~/repos/chesspy/yolo/weights/yolov4_tiny_chess_last.weights',
        help='Path to the yolo weights file.'
    )

    parser.add_argument(
        '--meta-path', type=str,
        default='~/repos/chesspy/yolo/data/chess_obj.data',
        help='Path to the yolo .data file.'
    )

    parser.add_argument(
        '--select-board', action='store_true',
        help='Interface for selecting the board outline in the video.'
    )

    parser.add_argument(
        '--show-best-move', action='store_true',
        help='Highlights best moves in the 2D board.'
    )

    parser.add_argument(
        '--save-video', action='store_true',
        help='Save the videos feed with the 2D board detection.'
    )

    parser.add_argument(
        '--debug', action='store_true',
        help='Shows debug window.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    cap = cv.VideoCapture(args.board_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    if not cap.isOpened():
        raise Exception('Error opening video stream!')

    if args.save_video:
        out_writer = cv.VideoWriter(
            'result.mp4',
            cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            30, (3000, 1080)
        )

    if args.select_board:
        ret, frame = cap.read()

        if not ret:
            raise Exception('Error reading the video stream!')

        roi_selector(frame)

    try:
        board_coords = np.array(
            load_json(open('./assets/board_pos.json'))['board_coords'],
            dtype='float32'
        )
    except FileNotFoundError:
        raise Exception('Board selection configuration not found! Run the script with the `--select-board` flag.')

    warp_width, warp_height = 400, 400
    warped_coords = np.array(
        [
            [0, 0],
            [warp_width - 1, 0],
            [warp_width - 1, warp_height - 1],
            [0, warp_height - 1]
        ], 
        dtype='float32'
    )

    perspective_matrix = cv.getPerspectiveTransform(board_coords, warped_coords)

    detector = Yolo(args)

    ret, frame = cap.read()

    # Create a mask for the ROI
    mask = np.zeros(frame.shape, np.uint8)
    roi_points = np.array(board_coords, np.int32).reshape((-1, 1, 2))
    mask = cv.polylines(mask, [roi_points], True, (255, 255, 255), 2)
    mask2 = cv.fillPoly(mask.copy(), [roi_points], (255, 255, 255))

    # Apply the mask to the frame
    roi = cv.bitwise_and(mask2, frame)
    roi_warped = cv.warpPerspective(roi, perspective_matrix, (warp_width, warp_height))
    roi_warped_gray = cv.cvtColor(roi_warped, cv.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    canny_edges = cv.Canny(roi_warped_gray, 100, 150)
    edges_dilated = cv.dilate(canny_edges, np.ones((1, 1), dtype=np.uint8))

    # Apply Hough Line Transform
    lines = cv.HoughLinesP(
        edges_dilated,
        rho=1, theta=np.pi / 180,
        threshold=70,
        minLineLength=40, maxLineGap=70
    )

    centers = get_lines_intersections(lines)

    squares = get_squares_coords_dict(centers)

    if args.show_best_move:
        stockfish = engine.SimpleEngine.popen_uci(r'../assets/stockfish_16.1')

    result_win = 'Result'
    cv.namedWindow(result_win, cv.WINDOW_KEEPRATIO)
    cv.moveWindow(result_win, 0, 49)
    cv.resizeWindow(result_win, 1600, 666)

    if args.debug:
        debug_win = 'Debug'
        cv.namedWindow(debug_win, cv.WINDOW_KEEPRATIO)
        cv.moveWindow(debug_win, 1619, 49)
        cv.resizeWindow(debug_win, 636, 629)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            raise Exception('Error reading the video stream!')

        detections = detector.detect(frame)

        fen = detections_to_fen(detections, perspective_matrix, squares)

        if args.show_best_move:
            best_move_white = str(
                stockfish.play(
                    Board(f'{fen} w'),
                    engine.Limit(time=0.1)
                ).move
            )

            best_move_black = str(
                stockfish.play(
                    Board(f'{fen} b'),
                    engine.Limit(time=0.1)
                ).move
            )

            board_2d_data = Image.open(
                BytesIO(svg2png(
                    bytestring=svg.board(
                        Board(fen),
                        arrows=[
                            svg.Arrow(
                                parse_square(best_move_white[:2]),
                                parse_square(best_move_white[-2:])
                            ),
                            svg.Arrow(
                                parse_square(best_move_black[:2]),
                                parse_square(best_move_black[-2:]),
                                color='blue'
                            )
                        ]
                    ),
                    output_width=frame_height, 
                    output_height=frame_height
                ))
            ).convert('RGBA')

        else:
            board_2d_data = Image.open(
                BytesIO(svg2png(
                    bytestring=svg.board(Board(fen)),
                    output_width=frame_height, 
                    output_height=frame_height
                ))
            ).convert('RGBA')

        board_2d = cv.cvtColor(
            np.array(board_2d_data),
            cv.COLOR_RGBA2BGR
        )

        processed_frame = np.concatenate(
            (frame, board_2d),
            axis=1
        )

        if args.debug:

            mask = np.zeros(frame.shape, np.uint8)
            roi_points = np.array(board_coords, np.int32).reshape((-1, 1, 2))
            mask = cv.polylines(mask, [roi_points], True, (255, 255, 255), 2)
            mask2 = cv.fillPoly(mask.copy(), [roi_points], (255, 255, 255))

            roi = cv.bitwise_and(mask2, frame)
            debug_img = cv.warpPerspective(roi, perspective_matrix, (warp_width, warp_height))

            # Draw the squares intersections
            for idx, (cx, cy) in enumerate(centers):
                cx = np.round(cx).astype(int)
                cy = np.round(cy).astype(int)

                cv.circle(debug_img, (cx, cy), 3, (255, 0, 255), -1)

            # Draw squares and square names
            for square_name, (pt1, pt2) in squares.items():
                x1, y1 = pt1
                x2, y2 = pt2

                cv.rectangle(
                    debug_img,
                    (x1 + 6, y1 + 6),
                    (x2 - 6, y2 - 6),
                    (107, 104, 255), 1
                )

                x = round((x1 + x2) / 2)
                y = round((y1 + y2) / 2)
                cv.putText(
                    debug_img,
                    square_name,
                    (x - 10, y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255)
                )

            # Draw detections
            for detection in detections:
                x, y, _, _ = detection['coords']

                # Projection the points
                x_p, y_p = cv.perspectiveTransform(
                    np.array([[[x, y]]], dtype=np.float64),
                    perspective_matrix
                ).reshape(2, ).astype(int)

                cv.circle(
                    debug_img,
                    (x_p, y_p),
                    3,
                    (30, 189, 133),
                    -1
                )

                cv.putText(
                    debug_img,
                    get_fen_piece_name(detection['class_name']),
                    (x_p, y_p - 3),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (30, 189, 133),
                    2
                )

            cv.imshow(debug_win, debug_img)

        cv.imshow(result_win, processed_frame)

        if args.save_video:
            out_writer.write(processed_frame)

        key = cv.waitKey(10)
        # Esc or Q key pressed
        if (key == 27 or key == ord('q')):
            if args.show_best_move:
                stockfish.quit()

            if args.save_video:
                out_writer.release()

            cv.destroyAllWindows()
            cap.release()

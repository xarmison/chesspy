from chess import Board, svg, engine, parse_square
from argparse import Namespace, ArgumentParser
from json import load as load_json
from cairosvg import svg2png
from io import BytesIO
from PIL import Image
import numpy as np
import cv2 as cv

from helper_functions import segment_lines, find_intersection, cluster_points, \
    get_squares_coords_dict, detections_to_fen, get_fen_piece_name

from board_selector import roi_selector
from yolo import Yolo


def parse_args() -> Namespace:
    """
        Parses command line arguments for the board detection script.

        Returns:
            argparse.Namespace: Parsed command line arguments.
    """

    parser = ArgumentParser(
        description='Identifies a chess position from an image.'
    )

    parser.add_argument(
        'board_image', type=str,
        help='Path to the image file to be processed.'
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

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    board = cv.imread(args.board_image)
    board_height, board_width, _ = board.shape

    if args.select_board:
        roi_selector(board)

    try:
        board_coords = np.array(
            load_json(open('./assets/board_pos.json'))['board_coords'],
            dtype='float32'
        )
    except FileNotFoundError:
        raise Exception('Board selection configuration not found! Run the script with the `--select-board` flag.')

    width = 400
    height = 400

    dest_coords = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype='float32')

    perspective_matrix = cv.getPerspectiveTransform(board_coords, dest_coords)

    mask = np.zeros(board.shape, np.uint8)
    points = np.array(board_coords, np.int32).reshape((-1, 1, 2))
    mask = cv.polylines(mask, [points], True, (255, 255, 255), 2)
    mask2 = cv.fillPoly(mask.copy(), [points], (255, 255, 255))

    roi = cv.bitwise_and(mask2, board)

    roi_warped = cv.warpPerspective(roi, perspective_matrix, (width, height))
    roi_warped_gray = cv.cvtColor(roi_warped, cv.COLOR_BGR2GRAY)

    canny_edges = cv.Canny(roi_warped_gray, 100, 150)
    dilated = cv.dilate(canny_edges, np.ones((1, 1), dtype=np.uint8))

    lines = cv.HoughLinesP(
        dilated, 
        rho=1, theta=np.pi/180, 
        threshold=70,  
        minLineLength=40, maxLineGap=70
    )

    # Segment the lines
    delta = 10
    h_lines, v_lines = segment_lines(lines, delta)

    # draw the segmented lines
    hough_img = roi_warped.copy()
    for line in h_lines:
        for x1, y1, x2, y2 in line:
            # Horizontal lines are red
            cv.line(
                hough_img, 
                (x1, y1), (x2, y2), 
                color=(0, 0, 255), thickness=1
            )

    for line in v_lines:
        for x1, y1, x2, y2 in line:
            # Vertical lines are blue
            cv.line(
                hough_img, 
                (x1, y1), (x2, y2), 
                color=(255, 0, 0), thickness=1
            )

    # find the line intersection points
    Px = []
    Py = []
    for h_line in h_lines:
        for v_line in v_lines:
            px, py = find_intersection(h_line, v_line)
            Px.append(px)
            Py.append(py)

    # Use clustering to find the centers of the data clusters
    P = np.float32(np.column_stack((Px, Py)))
    centers = cluster_points(P, 81)

    centers = centers[np.lexsort((centers[:, 0], centers[:, 1]))]

    # Draw the center of the clusters
    for idx, (cx, cy) in enumerate(centers):
        cx = np.round(cx).astype(int)
        cy = np.round(cy).astype(int)

        cv.circle(roi_warped, (cx, cy), radius=3, color=(255, 0, 255), thickness=-1)
    
    squares = get_squares_coords_dict(centers)

    # Draw squares and square names
    for square_name, (pt1, pt2) in squares.items():
        x1, y1 = pt1
        x2, y2 = pt2

        cv.circle(roi_warped, pt1, 2, (255, 255, 255), -1)

        cv.rectangle(
            roi_warped, 
            (x1 + 6, y1 + 6), 
            (x2 - 6, y2 - 6), 
            (107, 104, 255), 1
        )

        x = round((x1 + x2) / 2)
        y = round((y1 + y2) / 2)
        cv.putText(
            roi_warped,
            square_name,
            (x - 10, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255)
        )

    detector = Yolo(args)

    detections = detector.detect(board)

    # Draw detections
    for detection in detections:
        x, y, _, _ = detection['coords']

        # Projection the points
        x_p, y_p = cv.perspectiveTransform(
            np.array([[[x, y]]], dtype=np.float64),
            perspective_matrix
        ).reshape(2,).astype(int)

        cv.circle(
            roi_warped, 
            (x_p, y_p), 
            5, 
            detector.class_colors[detection['class_name']], 
            -1
        )

        cv.putText(
            roi_warped,
            get_fen_piece_name(detection['class_name']),
            (x_p, y_p - 3),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (30, 189, 133),
            2
        )

    fen = detections_to_fen(detections, perspective_matrix, squares)

    if args.show_best_move:
        stockfish = engine.SimpleEngine.popen_uci(r'../assets/stockfish_16.1')
        result = str(stockfish.play(Board(fen), engine.Limit(time=0.1)).move)
        stockfish.quit()

        board_2d_data = Image.open(
            BytesIO(svg2png(
                bytestring=svg.board(
                    Board(fen),
                    arrows=[svg.Arrow(
                        parse_square(result[:2]), 
                        parse_square(result[-2:])
                    )]
                ),
                output_width=board_height, output_height=board_height
            ))
        ).convert('RGBA')

    else:
        board_2d_data = Image.open(
            BytesIO(svg2png(
                bytestring=svg.board(Board(fen)),
                output_width=board_height, output_height=board_height
            ))
        ).convert('RGBA')

    board_2d = cv.cvtColor(
        np.array(board_2d_data), 
        cv.COLOR_RGBA2BGR
    )

    detections_img = np.concatenate(
        (
            detector.draw_detection_boxes(board, detections), 
            board_2d
        ), 
        axis=1
    )
    
    result_win = 'Result_1'
    cv.namedWindow(result_win, cv.WINDOW_KEEPRATIO)
    cv.moveWindow(result_win, 80, 60)
    cv.resizeWindow(result_win, 1351, 524)

    result_win_2 = 'Result_2'
    cv.namedWindow(result_win_2, cv.WINDOW_KEEPRATIO)
    cv.moveWindow(result_win_2, 1435, 60)
    cv.resizeWindow(result_win_2, 636, 629)

    cv.imshow(result_win, detections_img)
    cv.imshow(result_win_2, roi_warped)

    cv.waitKey(0)
    cv.destroyAllWindows()

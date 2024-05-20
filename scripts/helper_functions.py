from more_itertools import run_length
import numpy as np
import cv2 as cv


def segment_lines(lines: np.array, delta: int) -> tuple:
    """
        Segments the given lines into horizontal and vertical lines based on the specified delta value.

        Args:
            lines (np.array): Array of lines represented as (x1, y1, x2, y2) coordinates.
            delta (int): Threshold value to determine if a line is horizontal or vertical.

        Returns:
            tuple: A tuple containing two lists: horizontal lines and vertical lines.
    """

    h_lines = []
    v_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # x-values are near; line is vertical
            if abs(x2 - x1) < delta:  
                v_lines.append(line)

            # y-values are near; line is horizontal
            elif abs(y2 - y1) < delta:  
                h_lines.append(line)

    return h_lines, v_lines


def find_intersection(line1: np.array, line2: np.array) -> tuple:
    """
        Finds the intersection point of two lines.

        Args:
            line1 (np.array): The coordinates of the first line in the format [[x1, y1, x2, y2]].
            line2 (np.array): The coordinates of the second line in the format [[x3, y3, x4, y4]].

        Returns:
            tuple: The coordinates of the intersection point (intersect_x, intersect_y).
    """
    
    # extract points
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    # compute determinant
    intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / \
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

    intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / \
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

    return intersect_x, intersect_y


def cluster_points(points: np.array, nclusters: int) -> np.array:
    """
        Cluster the given points into the specified number of clusters using K-means algorithm.

        Args:
            points (np.array): The input array of points to be clustered.
            nclusters (int): The number of clusters to create.

        Returns:
            np.array: The array of cluster centers.

    """
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv.kmeans(
        points, nclusters, None,
        criteria, 10, cv.KMEANS_PP_CENTERS
    )

    return centers


def get_lines_intersections(lines: np.array) -> np.array:
    """
        Find the intersection points of horizontal and vertical lines.

        Args:
            lines (np.array): Array of lines represented as (rho, theta) pairs.

        Returns:
            np.array: Array of intersection points represented as (x, y) pairs.
    """
    
    # Segment the lines into horizontal and vertical lines
    h_lines, v_lines = segment_lines(lines, delta=10)

    # find the line intersection points 
    insections_x = []
    insections_y = []
    for h_line in h_lines:
        for v_line in v_lines:
            insection_x, intersection_y = find_intersection(h_line, v_line)
            insections_x.append(insection_x)
            insections_y.append(intersection_y)

    # Clustering the centers of the intersectiosn
    insections_pairs = np.float32(np.column_stack((insections_x, insections_y)))
    centers = cluster_points(insections_pairs, nclusters=81)

    # Sort the centers by x and y
    return centers[np.lexsort((centers[:, 0], centers[:, 1]))]


def get_squares_coords_dict(centers: np.array) -> dict:
    """
        Returns a dictionary containing the coordinates of the squares on a chessboard.

        Parameters:
        centers (np.array): An array containing the coordinates of the centers of the squares.

        Returns:
        dict: A dictionary where the keys are the square positions in algebraic notation (e.g., 'a1', 'b2', etc.)
            and the values are tuples containing the coordinates of the square's center and the
            coordinates of the next square's center.
    """

    squares = {}
    edges = [8, 17, 26, 35, 44, 53, 62, 71]

    columnidx = 0
    column = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    rank = 8

    for idx, center in enumerate(centers):
        if idx not in edges:
            squares[f'{column[columnidx]}{rank}'] = (
                np.round(center).astype(int),
                np.round(centers[idx + 10]).astype(int)
            )

            if columnidx < 7:
                columnidx += 1
            else:
                columnidx = 0
                rank -= 1

        elif idx == 71:
            break

        else:
            continue

    return squares


def fen_from_board(board_matrix: np.array) -> str:
    """
        Converts a chess board matrix into a FEN (Forsyth-Edwards Notation) string.

        Args:
            board_matrix (np.array): The chess board matrix.

        Returns:
            str: The FEN string representation of the chess board.

    """
    def convert_rank(rank):
        """
            Converts a chess rank to a simplified representation.

            Args:
                rank (str): The chess rank to be converted.

            Returns:
                str: The simplified representation of the chess rank.
        """

        return ''.join(
            value * count if value else str(count)
            for value, count in run_length.encode(rank)
        )

    return '/'.join(map(convert_rank, board_matrix))


def get_fen_piece_name(piece: str) -> str:
    """
        Converts a piece name to its corresponding FEN notation.

        Args:
            piece (str): The name of the piece.

        Returns:
            str: The FEN notation of the piece.

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

    return fen_piece_map.get(piece, '')


def detections_to_fen(detections: list, perspective_matrix: np.array, squares: dict) -> str:
    """
        Convert a list of detections to a FEN (Forsyth-Edwards Notation) string representation of a chessboard.

        Args:
            detections (list): A list of detections containing information about the detected chess pieces.
            perspective_matrix (np.array): The perspective transformation matrix used to project the points.
            squares (dict): A dictionary containing the coordinates of the squares on the chessboard.

        Returns:
            str: The FEN string representation of the chessboard.

    """

    # Iterate over each detection
    for detection in detections:
        x, y, _, _ = detection['coords']

        # Project the points
        x_p, y_p = cv.perspectiveTransform(
            np.array([[[x, y]]], dtype=np.float64),
            perspective_matrix
        ).reshape(2, )

        detection['square'] = '  '

        # Iterate over each square
        for square_name, (top_left, bottom_right) in squares.items():
            x1, y1 = top_left
            x2, y2 = bottom_right

            # Check if the middle point of the bounding box is in the square
            if x1 < x_p < x2 and y1 < y_p < y2:
                detection['square'] = square_name

    matrix_map = {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3,
        'e': 4,
        'f': 5,
        'g': 6,
        'h': 7,
        '1': 7,
        '2': 6,
        '3': 5,
        '4': 4,
        '5': 3,
        '6': 2,
        '7': 1,
        '8': 0,
    }

    board_matrix = np.empty([8, 8], dtype=str)

    # Iterate over each detection
    for detection in detections:
        row = matrix_map.get(detection['square'][0], None)
        column = matrix_map.get(detection['square'][1], None)

        if row is None or column is None:
            continue

        board_matrix[column][row] = get_fen_piece_name(detection['class_name'])

    return fen_from_board(board_matrix)


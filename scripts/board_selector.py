from json import dump as save_json
from typing import Any
import numpy as np
import cv2 as cv

def draw_roi(event: int, x: int, y: int, flags: int, params: Any | None) -> None:
    """
        Callback function for mouse events used to draw a region of interest (ROI) on an image.

        Args:
            event (int): The type of mouse event.
            x (int): The x-coordinate of the mouse event.
            y (int): The y-coordinate of the mouse event.
            flags (int): Additional flags for the mouse event.
            params (Any | None): Additional parameters passed to the callback function.

        Returns:
            None
    """

    frame, pts, roi_win = params

    circle_radius = int(min(frame.shape[:2]) * 0.01)
    line_tickness = int(min(frame.shape[:2]) * 0.003)

    img2 = frame.copy()

    if event == cv.EVENT_LBUTTONDOWN:
        pts.append((x, y))

    if event == cv.EVENT_RBUTTONDOWN:
        pts.pop()

    if event == cv.EVENT_MBUTTONDOWN:
        mask = np.zeros(img2.shape, np.uint8)

        points = np.array(pts, np.int32).reshape((-1, 1, 2))

        mask = cv.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv.fillPoly(mask.copy(), [points], (255, 255, 255))
        mask3 = cv.fillPoly(mask.copy(), [points], (0, 255, 0))

        show_image = cv.addWeighted(
            src1=img2, alpha=0.8, src2=mask3, beta=0.2, gamma=0
        )

        cv.imshow('Selection Mask', show_image)

        selections_win = 'Selected ROI'
        cv.namedWindow(selections_win, cv.WINDOW_KEEPRATIO)

        cv.imshow('ROI', selections_win)
        cv.waitKey(0)

    if len(pts) > 0:
        cv.circle(img2, pts[-1], circle_radius + 3, (107, 104, 255), -1)

    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv.circle(img2, pts[i], circle_radius, (107, 104, 255), -1)
            cv.line(
                img=img2, pt1=pts[i], pt2=pts[i + 1],
                color=(198, 220, 132), thickness=line_tickness
            )

    cv.imshow(roi_win, img2)


def roi_selector(frame: np.array) -> None:
    """
        Allows the user to select the outline of the chessboard region of interest (ROI) on the given frame.

        Args:
            frame (np.array): The input frame on which the ROI is to be selected.

        Returns:
            None, saves the selected ROI to a JSON file at `./assets/board_pos.json`.
    """
    
    print(
        'Select the outline of the chessboard following the sequence:\n' +
        '[start]                             \n' +
        'top left ---------------> top right \n' +
        '   ^                          |     \n' +
        '   |                          |     \n' +
        '   |                          |     \n' +
        '   |                          ⌄     \n' +
        'bottom left <------------ bottom right\n' +
        'Close the shape by selecting the first point.\n' +
        'Press `s` or `enter` to save the shape, and `esc` to cancel.'
    )

    roi_win = 'Board Selection'
    cv.namedWindow(roi_win, cv.WINDOW_KEEPRATIO)
    cv.moveWindow(roi_win, 473, 64)
    cv.resizeWindow(roi_win, 1224, 736)

    cv.imshow(roi_win, frame)

    pts = []
    cv.setMouseCallback(roi_win, draw_roi, (frame, pts, roi_win))

    while True:
        key = cv.waitKey(1) & 0xFF
        # Esc key
        if key == 27:
            break

        # `s` or enter key
        if key == ord('s') or key == 13:
            pts.pop()
            roi = {'board_coords': pts}

            save_path = './assets/board_pos.json'

            with open(save_path, 'w') as save_file:
                save_json(
                    roi, save_file,
                    ensure_ascii=False, indent=4
                )

            print(f'[I] Board selection configuration saved to `{save_path}`')

            break

    cv.destroyWindow(roi_win)


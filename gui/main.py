from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, QObject
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QImage, QPixmap
from typing import Any

from chess import Board, svg, engine, parse_square
from json import load as load_json
from cairosvg import svg2png
from os.path import isfile
from io import BytesIO
import mediapipe as mp
from PIL import Image
import sys

from yolo import Yolo
import numpy as np
import cv2 as cv

sys.path.append('../')
from scripts.board_selector import roi_selector
from scripts.helper_functions import get_lines_intersections, get_squares_coords_dict, \
    detections_to_fen, get_fen_piece_name


class FrameProcessor:
    """
        Class for processing frames of a chess game video.
    """

    def __init__(self, options: dict) -> None:
        """
            Initializes a FrameProcessor object.

            Args:
                options (dict): Options for frame processing.
        """

        self.file_name = None

        self.options = options
        self.set_options(self.options)

        self.cap = None
        self.frame_idx = 0
        
        # Stores the previous board pos
        self.prev_board = None

        # Hand detection
        self.hand_model = mp.solutions.hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def set_options(self, options: dict) -> None:
        """
            Sets the options for frame processing.

            Args:
                options (dict): Options for frame processing.
        """

        if self.file_name is None:
            return

        self.options = options

        if self.options['show_best_move']:
            self.load_stockfish()
        
        # Creates a stream object for writing the output
        if self.options['save_video']:
            result_file_name =  f'./results/{self.file_name}_result.mp4'

            self.out_writer = cv.VideoWriter(
                result_file_name,
                cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                30, 
                (self.frame_width + self.frame_height, self.frame_height)
            )

    def load_video(self, file_path: str) -> None:
        """
            Loads a video file for processing.

            Args:
                file_path (str): The path to the video file.
        """

        self.file_name = file_path.split('/')[-1].split('.')[0]

        self.cap = cv.VideoCapture(file_path)
        self.frame_width = int(self.cap.get(3)) 
        self.frame_height = int(self.cap.get(4))

        if not self.cap.isOpened():
            raise Exception('Error opening video stream!')

        self.setup()

    def setup(self) -> None:
        """
            Performs the initial setup for frame processing.
        """

        if self.options['select_board']:
            ret, frame = self.cap.read()

            if not ret:
                raise Exception('Error reading the video stream!')

            roi_selector(frame)

        if isfile('./assets/board_pos.json'):
            board_coords = np.array(
                load_json(open('./assets/board_pos.json'))['board_coords'],
                dtype='float32'
            )
        else:
            raise Exception('Create board selection configuration first!')

        self.warp_width = 400
        self.warp_height = 400

        warped_coords = np.array([
            [0, 0],
            [self.warp_width - 1, 0],
            [self.warp_width - 1, self.warp_height - 1],
            [0, self.warp_height - 1]
        ], dtype='float32')

        self.perspective_matrix = cv.getPerspectiveTransform(
            board_coords, 
            warped_coords
        )

        self.detector = Yolo()

        ret, frame = self.cap.read()
        self.prev_frame = frame.copy()

        mask = np.zeros(frame.shape, np.uint8)
        roi_points = np.array(board_coords, np.int32).reshape((-1, 1, 2))
        mask = cv.polylines(mask, [roi_points], True, (255, 255, 255), 2)
        self.mask2 = cv.fillPoly(mask.copy(), [roi_points], (255, 255, 255))

        roi = cv.bitwise_and(self.mask2, frame)

        roi_warped = cv.warpPerspective(
            roi, self.perspective_matrix, 
            (self.warp_width, self.warp_height)
        )
        roi_warped_gray = cv.cvtColor(roi_warped, cv.COLOR_BGR2GRAY)

        canny_edges = cv.Canny(roi_warped_gray, 100, 150)
        edges_dilated = cv.dilate(canny_edges, np.ones((1, 1), dtype=np.uint8))
        
        lines = cv.HoughLinesP(
            edges_dilated,
            rho=1, theta=np.pi/180, 
            threshold=70,  
            minLineLength=40, maxLineGap=70
        )

        self.centers = get_lines_intersections(lines)

        self.squares = get_squares_coords_dict(self.centers)

        if self.options['show_best_move']:
            self.load_stockfish()

        if self.options['debug']:
            self.debug_win = 'Debug'
            cv.namedWindow(self.debug_win, cv.WINDOW_KEEPRATIO)
            cv.moveWindow(self.debug_win, 1619, 49)
            cv.resizeWindow(self.debug_win, 636, 629)

    def load_stockfish(self) -> None:
        """
            Loads the Stockfish chess engine.
        """

        self.stockfish = engine.SimpleEngine.popen_uci(r'../assets/stockfish_16.1')

    def process_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """
            Processes a single frame of the video.

            Returns:
                frame (np.ndarray): The original video frame.
                board_2d (np.ndarray): The 2D representation of the chessboard.
        """

        if self.cap is None:
            print('[I] Load the video file first.')
            return None

        ret, frame = self.cap.read()

        if not ret:
            raise Exception('Error reading the video stream!')
        
        hand_result = self.hand_model.process(
            cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        )

        # If there is a hand in frame it should not be processed by yolo
        hand_in_frame = hand_result.multi_hand_landmarks is not None
        
        # Control the frequency in which the yolo is executed
        yolo_interval = self.frame_idx % self.options['run_yolo_every']

        if hand_in_frame or yolo_interval:
            self.frame_idx += 1
            return frame, self.prev_board

        detections = self.detector.detect(frame)

        fen = detections_to_fen(detections, self.perspective_matrix, self.squares)

        if self.options['show_best_move']:
            best_move_white = str(self.stockfish.play(
                Board(f'{fen} w'), 
                engine.Limit(time=0.1)
            ).move)

            best_move_black = str(self.stockfish.play(
                Board(f'{fen} b'), 
                engine.Limit(time=0.1)
            ).move)

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
                    output_width=self.frame_height, 
                    output_height=self.frame_height
                ))
            ).convert('RGBA')

        else :
            board_2d_data = Image.open(
                BytesIO(svg2png(
                    bytestring=svg.board(Board(fen)),
                    output_width=self.frame_height, 
                    output_height=self.frame_height
                ))
            ).convert('RGBA')
        
        board_2d = cv.cvtColor(
            np.array(board_2d_data), 
            cv.COLOR_RGBA2BGR
        )

        self.prev_board = board_2d

        if self.options['debug']:

            roi = cv.bitwise_and(self.mask2, frame)

            debug_img = cv.warpPerspective(
                roi, 
                self.perspective_matrix, 
                (self.warp_width, self.warp_height)
            )

            # Draw the squares intersections
            for (cx, cy) in self.centers:
                cx = np.round(cx).astype(int)
                cy = np.round(cy).astype(int)

                cv.circle(debug_img, (cx, cy), 3, (255, 0, 255), -1)

            # Draw squares and square names
            for square_name, (pt1, pt2) in self.squares.items():
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
                    self.perspective_matrix
                ).reshape(2,).astype(int)

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

            cv.imshow('Debug', debug_img)
        
        self.frame_idx += 1

        return frame, board_2d

class Thread(QThread):
    """
        A custom QThread class for processing video frames and emitting signals.
    """

    video_pixmap = pyqtSignal(QImage)
    board_pixmap = pyqtSignal(QImage)

    def __init__(self, parent: QObject | None = None) -> None:
        """
            Initializes the Thread object.

            Args:
                parent (QObject): The parent object of the Thread.

            Returns:
                None
        """

        QThread.__init__(self, parent=parent)

        self.paused = False

        self.options = {
            'select_board': False,
            'show_best_move': False,
            'save_video': False,
            'run_yolo_every': 10,
            'debug': False
        }

        self.processor = FrameProcessor(self.options)

        self.set_placeholders()

    def play_pause(self) -> None:
        """
            Toggles the paused flag.

            Args:
                None

            Returns:
            None
        """

        self.paused = not self.paused

    def send_image_signals(self, video_frame: np.array, board_image: np.array) -> None:
        """
            Sends the video and board image signals.

            Args:
                video_frame (np.array): The video frame as a numpy array.
                board_image (np.array): The board image as a numpy array.

            Returns:
                None
        """

        video_view_width, video_view_height = 1024, 576
        bytes_per_line_video = 3 * video_view_width

        video_frame = cv.cvtColor(cv.resize(
            video_frame, 
            (video_view_width, video_view_height),
            cv.INTER_LINEAR
        ), cv.COLOR_BGR2RGB)

        board_view_width, board_view_height = 576, 576
        bytes_per_line_board = 3 * board_view_width

        board_image = cv.cvtColor(cv.resize(
            board_image, 
            (board_view_width, board_view_height),
            cv.INTER_LINEAR
        ), cv.COLOR_BGR2RGB)

        self.video_pixmap.emit(QImage(
            video_frame.data, 
            video_view_width, video_view_height,
            bytes_per_line_video,
            QtGui.QImage.Format.Format_RGB888
        ))

        self.board_pixmap.emit(QImage(
            board_image.data, 
            board_view_width, board_view_height,
            bytes_per_line_board,
            QtGui.QImage.Format.Format_RGB888
        ))

    def set_placeholders(self) -> None:
        """
            Sets the initial image placeholders.

            Args:
                None

            Returns:
                None
        """

        video_placeholder = cv.imread(
            './assets/video_placeholder.png',
            cv.COLOR_BGR2RGB
        )

        board_placeholder = cv.imread(
            './assets/board_placeholder.png',
            cv.COLOR_BGR2RGB
        )

        self.send_image_signals(video_placeholder, board_placeholder)

    def run(self) -> None:
        """
            Runs the thread and continuously processes video frames.

            Args:
                None

            Returns:
                None
        """

        while True:
            if not self.paused:
                continue

            video_frame, board_image = self.processor.process_frame()

            self.send_image_signals(video_frame, board_image)

class MainWindow(QMainWindow):
    """
        The main window of the ChessPy application.
    """

    def __init__(self) -> None:
        """
            Initializes the MainWindow object.

            Args:
                None

            Returns:
                None
        """

        super(MainWindow, self).__init__()

        # Internal variavles
        self.video_path = ''
        self.paused = True

        self.th = Thread(self)

        self.setup_ui(self)

        self.th.video_pixmap.connect(self.set_frame)
        self.th.board_pixmap.connect(self.set_board)
        
        self.th.set_placeholders()
        self.th.start()

        self.show()

    def setup_ui(self, MainWindow: QMainWindow) -> None:
        """
            Set up the user interface for the main window.

            Args:
                MainWindow (QMainWindow): The main window object.

            Returns:
                None
        """

        MainWindow.setObjectName('MainWindow')
        MainWindow.resize(1760, 594)
        MainWindow.setWindowIcon(QtGui.QIcon('assets/icon.png'))

        self.allFather = QtWidgets.QWidget(MainWindow)
        self.allFather.setObjectName('allFather')
        
        self.gridLayout = QtWidgets.QGridLayout(self.allFather)
        self.gridLayout.setObjectName('gridLayout')

        self.sideBar = QtWidgets.QVBoxLayout()
        self.sideBar.setObjectName('sideBar')

        self.fileSettings = QtWidgets.QGridLayout()
        self.fileSettings.setObjectName('fileSettings')

        self.open_file_btn = QtWidgets.QPushButton(self.allFather)
        self.open_file_btn.setObjectName('open_file_btn')
        self.open_file_btn.clicked.connect(self.open_file)
        self.fileSettings.addWidget(self.open_file_btn, 1, 0, 1, 1)

        self.sideBar.addLayout(self.fileSettings)
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, 
            QtWidgets.QSizePolicy.Policy.Minimum, 
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.sideBar.addItem(spacerItem)

        self.opitionalSettings = QtWidgets.QGridLayout()
        self.opitionalSettings.setObjectName('opitionalSettings')

        self.line = QtWidgets.QFrame(self.allFather)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setObjectName('line')
        self.opitionalSettings.addWidget(self.line, 3, 0, 1, 1)

        # Checkboxes for optional settings
        self.optional_settings_label = QtWidgets.QLabel(self.allFather)
        self.optional_settings_label.setObjectName('optional_settings_label')
        self.opitionalSettings.addWidget(self.optional_settings_label, 2, 0, 1, 1)

        self.select_board = QtWidgets.QCheckBox(self.allFather)
        self.select_board.setObjectName('select_board')
        self.select_board.stateChanged.connect(lambda: self.change_options('select_board'))
        self.opitionalSettings.addWidget(self.select_board, 4, 0, 1, 1)

        self.show_best_move = QtWidgets.QCheckBox(self.allFather)
        self.show_best_move.setObjectName('show_best_move')
        self.show_best_move.stateChanged.connect(lambda: self.change_options('show_best_move'))
        self.opitionalSettings.addWidget(self.show_best_move, 5, 0, 1, 1)

        self.save_video = QtWidgets.QCheckBox(self.allFather)
        self.save_video.setObjectName('save_video')
        self.save_video.stateChanged.connect(lambda: self.change_options('save_video'))
        self.opitionalSettings.addWidget(self.save_video, 8, 0, 1, 1)

        self.debug = QtWidgets.QCheckBox(self.allFather)
        self.debug.setObjectName('debug')
        self.debug.stateChanged.connect(lambda: self.change_options('debug'))
        self.opitionalSettings.addWidget(self.debug, 9, 0, 1, 1)

        self.sideBar.addLayout(self.opitionalSettings)
        spacerItem1 = QtWidgets.QSpacerItem(
            20, 40, 
            QtWidgets.QSizePolicy.Policy.Minimum, 
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.sideBar.addItem(spacerItem1)

        self.limiter_layout = QtWidgets.QGridLayout()
        self.limiter_layout.setObjectName('limiter_layout')
        self.sideBar.addLayout(self.limiter_layout)

        self.limiter_label = QtWidgets.QLabel(self.allFather)
        self.limiter_label.setObjectName('limiter_label')
        self.limiter_layout.addWidget(self.limiter_label, 2, 0, 1, 1)

        self.line_2 = QtWidgets.QFrame(self.allFather)
        self.line_2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_2.setObjectName("line_2")
        self.limiter_layout.addWidget(self.line_2, 3, 0, 1, 1)

        self.limiter_number = QtWidgets.QSpinBox(self.allFather)
        self.limiter_number.setMinimum(1)
        self.limiter_number.setMaximum(100)
        self.limiter_number.setProperty("value", 10)
        self.limiter_number.setObjectName("limiter_number")
        self.limiter_layout.addWidget(self.limiter_number, 4, 0, 1, 1)

        self.limiter_btn = QtWidgets.QPushButton(self.allFather)
        self.limiter_btn.setObjectName("limiter_btn")
        self.limiter_btn.clicked.connect(self.change_yolo_limiter)
        self.limiter_layout.addWidget(self.limiter_btn, 5, 0, 1, 1)

        spacerItem2 = QtWidgets.QSpacerItem(
            20, 40, 
            QtWidgets.QSizePolicy.Policy.Minimum, 
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.sideBar.addItem(spacerItem2)

        self.mainBtns = QtWidgets.QGridLayout()
        self.mainBtns.setObjectName('mainBtns')

        self.play_btn = QtWidgets.QPushButton(self.allFather)
        self.play_btn.setObjectName('play_btn')
        self.play_btn.clicked.connect(self.th.play_pause)
        self.mainBtns.addWidget(self.play_btn, 1, 0, 1, 1)

        self.sideBar.addLayout(self.mainBtns)
        self.gridLayout.addLayout(self.sideBar, 1, 2, 1, 1)

        self.video_feed = QtWidgets.QLabel()
        self.video_feed.setObjectName('video_feed')
        self.gridLayout.addWidget(self.video_feed, 1, 0, 1, 1)

        self.board_feed = QtWidgets.QLabel()
        self.board_feed.setObjectName('board_feed')
        self.gridLayout.addWidget(self.board_feed, 1, 1, 1, 1)

        MainWindow.setCentralWidget(self.allFather)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1480, 30))
        self.menubar.setObjectName('menubar')

        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName('menuFile')
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName('statusbar')
        MainWindow.setStatusBar(self.statusbar)

        self.action_exit = QtGui.QAction(MainWindow)
        self.action_exit.setObjectName('action_exit')
        self.action_exit.triggered.connect(self.closeEvent)

        finish = QtGui.QAction("Quit", self)
        finish.triggered.connect(self.closeEvent)

        self.action_open = QtGui.QAction(MainWindow)
        self.action_open.setObjectName('action_open')
        self.action_open.triggered.connect(self.open_file)

        self.menuFile.addAction(self.action_open)
        self.menuFile.addAction(self.action_exit)

        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslate_ui(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslate_ui(self, MainWindow: QMainWindow) -> None:
        """
            Retranslates the user interface elements with translated text.

            Args:
                MainWindow (QMainWindow): The main window of the application.

            Returns:
                None
        """

        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate('MainWindow', 'ChessPy'))

        self.open_file_btn.setText(_translate('MainWindow', 'Open File'))
        self.open_file_btn.setShortcut(_translate('MainWindow', 'Ctrl+O'))

        self.optional_settings_label.setToolTip(_translate('MainWindow', 'Sets the additional settings.'))
        self.optional_settings_label.setText(_translate('MainWindow', 'Optional Settings'))

        self.save_video.setToolTip(_translate('MainWindow', 'Creates a video file with the analysis results.'))
        self.save_video.setText(_translate('MainWindow', 'Save Video'))

        self.show_best_move.setToolTip(_translate('MainWindow', 'Draws both axis found through PCA. '))
        self.show_best_move.setText(_translate('MainWindow', 'Show best Move'))

        self.select_board.setToolTip(_translate('MainWindow', 'Launches the interface to select the chessboard position.'))
        self.select_board.setText(_translate('MainWindow', 'Select Board'))

        self.debug.setToolTip(_translate('MainWindow', 'Launches the debuging window.'))
        self.debug.setText(_translate('MainWindow', 'Debug'))

        self.limiter_label.setText(_translate("MainWindow", "Run YOLO every x frames"))
        self.limiter_btn.setText(_translate("MainWindow", "Change"))

        self.play_btn.setToolTip(_translate('MainWindow', 'Resumes or pause the video stream.'))
        self.play_btn.setText(_translate('MainWindow', 'Play/Pause'))

        self.menuFile.setTitle(_translate('MainWindow', 'File'))

        self.action_exit.setText(_translate('MainWindow', 'Exit'))
        self.action_exit.setToolTip(_translate('MainWindow', 'Close application'))
        self.action_exit.setStatusTip(_translate('MainWindow', 'Close application'))
        self.action_exit.setWhatsThis(_translate('MainWindow', 'Close application'))
        self.action_exit.setShortcut(_translate('MainWindow', 'Ctrl+Q'))

        self.action_open.setText(_translate('MainWindow', 'Open Video'))
        self.action_open.setToolTip(_translate('MainWindow', 'Open video file for analysis'))
        self.action_open.setShortcut(_translate('MainWindow', 'Ctrl+O'))
    
    def closeEvent(self, event: Any) -> None:
        """
            Makes sure the user wants to close the application.

            Args:
                event (Any): The event object.

            Returns:
                None
        """

        choice = QtWidgets.QMessageBox.question(
            self, 'Exit Application', 'Are you sure ?', 
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )

        if choice == QtWidgets.QMessageBox.StandardButton.Yes:
            cv.destroyAllWindows()
            self.th.terminate()
            sys.exit()

    def open_file(self) -> None:
        """
            Opens a file dialog to select a video file.

            Args:
                None

            Returns:
                None
        """

        self.video_path, _ = QFileDialog.getOpenFileName(self)

        self.th.processor.load_video(self.video_path)

        self.statusBar().showMessage('Video Loaded Successfully! Press the play button to start.')

    def change_options(self, option: str) -> None:
        """
            Changes the options for frame processing.

            Args:
                option (str): The option to change.

            Returns:
                None
        """

        self.th.options[option] = not self.th.options[option] 

        self.th.processor.set_options(self.th.options)

    def change_yolo_limiter(self) -> None:
        """
            Changes the YOLO limiter value.

            Args:
                None

            Returns:
                None
        """

        self.th.options['run_yolo_every'] = self.limiter_number.value()

        self.th.processor.set_options(self.th.options)

    @pyqtSlot(QImage)
    def set_frame(self, frame: QImage) -> None:
        """
            Sets the video frame to the video feed.

            Args:
                frame (QImage): The video frame as a QImage.

            Returns:
                None
        """

        self.video_feed.setPixmap(QPixmap.fromImage(frame))

    @pyqtSlot(QImage)
    def set_board(self, board: QImage) -> None:
        """
            Sets the board image to the board feed.

            Args:
                board (QImage): The board image as a QImage.

            Returns:
                None
        """

        self.board_feed.setPixmap(QPixmap.fromImage(board))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWindow()

    sys.exit(app.exec())
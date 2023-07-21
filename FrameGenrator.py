from os.path import exists, isdir
from PIL import Image
import cv2


class FrameGenerator:
    def __init__(self, path: str, size_of_index=5, start='img', end='jpg') -> None:
        if not exists(path):
            raise FileExistsError("This file does not exist")

        self.t = 0  # the index of the frame. relevant for directory of frames
        self.path = path  # path to directory/video
        # size of the inedx as a string . relevant for directory of frames
        self.size_of_index = size_of_index
        self.end = end  # the end of the frames, 'jpg' for example. relevant for directory of frames
        # start of the names of the frames. 'img' for example. relevant for directory of frames
        self.start = start
        self.batch_size = 1  # the batch size. if working on 2 videos.
        self.is_dir = isdir(path)  # is the path a directory or not.
        if not self.is_dir:
            self.vidcap = cv2.VideoCapture(path)  # the video capture object

    def next_video(self):
        """
        return the next frame when the path is a path to a video.
        :params:
            None
        :returns:
            the next frame
        """
        ret, frame = self.vidcap.read()
        if ret:
            self.t += 1
            return Image.fromarray(frame)

    def next(self):
        """
        return the next frame.
        :params:
            None
        :returns:
            the next frame
        """
        if self.is_dir:
            return self.next_folder()
        else:
            return self.next_video()

    def next_folder(self):
        """
        return the next frame when the path is a path to a directory of frames.
        :params:
            None
        :returns:
            the next frame
        """
        temp_path = f'{self.path}/{self.start}_{str(self.t+1).zfill(self.size_of_index)}.{self.end}'
        if exists(temp_path):
            self.t += 1
            img = Image.open(temp_path)
            return img

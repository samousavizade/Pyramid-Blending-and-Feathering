from shutil import move
from tkinter.messagebox import NO
import cv2 as cv
import cv2
import numpy as np
import skimage.filters as filters
import matplotlib.pyplot as plt

class PolygonMaker:
    def __init__(self, points, mask_shape):
        self.points = points
        h, w, _ = mask_shape
        self.mask = np.zeros((h, w), np.uint8)

    class ClickHandler:
        image = None
        POINTS_SIZE = 0

        def __init__(self, image, window_name):
            self.image = image.copy()
            self.window_name = window_name
            cv.imshow(self.window_name, image)

            h, w, _ = self.image.shape
            self.counter = 0
            self.points = []
            print('clicked vertices of polygon:')

        def get_points(self):
            return np.array([[x, y] for x, y in self.points], np.int)

        def click_event(self, event, clicked_x, clicked_y, flags, params):
            if event == cv.EVENT_LBUTTONDOWN:
                print(f'Point {len(self.points) + 1} Coordinates is : {clicked_x}, {clicked_y}')
                point = np.array([clicked_x, clicked_y])
                cv.imshow(self.window_name, self.image)
                self.points.append(point)

    def get_filled_polygon(self):
        return cv.fillConvexPoly(self.mask, self.points, 255)


class PolygonMover:
    MAX_INTENSITY = 255

    def __init__(self, target, polygon, window_name):
        self.window_name = window_name
        self.target = target
        self.target_copy = self.target.copy()

        h, w, _ = self.target.shape
        self.polygon = np.zeros((h, w), np.uint8)
        self.polygon[np.where(polygon != 0)] = 255

        self.polygon_copy = self.polygon.copy()

        self.mouse_is_moving = False
        self.is_first_click = True
        self.x_before_move, self.y_before_move = 0, 0
        self.x_last_move, self.y_last_move = 0, 0

    @staticmethod
    def translate(input, delta_x, delta_y):
        translation_matrix = np.float32([[1, 0, delta_x],
                                         [0, 1, delta_y]])

        h, w = input.shape
        return cv.warpAffine(input, translation_matrix, (w, h))

    def mouse_moving_handler(self, event, current_x, current_y, flags, param):
        if event == cv.EVENT_LBUTTONUP:
            self.mouse_is_moving = False

        elif event == cv.EVENT_LBUTTONDOWN:
            self.mouse_is_moving = True
            self.x_last_move, self.y_last_move = current_x, current_y
            if self.is_first_click:
                self.x_before_move, self.y_before_move = current_x, current_y
                self.is_first_click = False

        elif event == cv.EVENT_MOUSEMOVE:
            if self.mouse_is_moving:
                delta_x = current_x - self.x_last_move
                delta_y = current_y - self.y_last_move
                self.polygon_copy = PolygonMover.translate(
                    self.polygon_copy, delta_x, delta_y)
                self.x_last_move, self.y_last_move = current_x, current_y

    def get_current_frame(self):
        current_frame = self.target.copy()
        current_frame[self.polygon_copy != 0] = 255
        return current_frame

    def get_moved_polygon(self):
        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name,
                            self.mouse_moving_handler)

        while True:
            current_frame = self.get_current_frame()
            cv.imshow(self.window_name, current_frame)

            pressed_key = cv.waitKey(1)
            if pressed_key == ord('c'):
                break

        cv.destroyAllWindows()

        total_delta_x, total_delta_y = self.x_last_move - \
            self.x_before_move, self.y_last_move - self.y_before_move
        return (total_delta_x, total_delta_y), self.polygon_copy


class PyramidBlender:
    class PyramidBuilder:

        def __init__(self, image, levels_number):
            self.image = image
            self.levels_number = levels_number

        def construct_gaussian_pyramid(self):
            # build gaussian pyramid
            current = self.image.copy()
            pyramid = [current]
            for _ in range(self.levels_number):
                current = cv.pyrDown(current)
                pyramid.append(current)

            return pyramid

        def construct_laplacian_pyramid(self):
            # build laplacian pyramid

            gp = self.construct_gaussian_pyramid()
            pyramid = [gp[self.levels_number - 1]]
            for i in range(self.levels_number - 1, 0, -1):
                gu = cv.pyrUp(gp[i])
                current = cv.subtract(gp[i - 1], gu)

                pyramid.append(current)

            return pyramid

    def __init__(self, image1, image2, mask, levels_number):
        self.image1 = image1
        self.CHANNEL_SIZE = self.image1.shape[2]

        self.image2 = image2
        self.height, self.width, _ = self.image2.shape

        self.mask = mask
        self.levels_number = levels_number

    @staticmethod
    def construct_blended_pyramid(L1, L2, R, n_level):
        blended_pyramid = []
        # calculate blended level based on formula
        for i in range(len(L1)):
            blend = R[i] * L1[n_level - 1 - i] + (1 - R[i]) * L2[n_level - 1 - i]
            blended_pyramid.append(blend)

        return blended_pyramid

    @staticmethod
    def reconstruct_channel(blended_pyramid):
        blended_channel = blended_pyramid[-1]
        h, w = blended_channel.shape
        # reconstruct channel based from last level of blended pyramid
        for i in reversed(range(0, len(blended_pyramid) - 1)):
            # up sampling
            pyr_up = cv.pyrUp(blended_channel)
            # check odd column and rows
            current_level_h, current_level_w = blended_pyramid[i].shape
            pyr_up = pyr_up[:-1, :] if h * 2 > current_level_h else pyr_up
            pyr_up = pyr_up[:, :-1] if w * 2 > current_level_w else pyr_up
            # add to last level
            blended_channel = pyr_up + blended_pyramid[i]

        return blended_channel

    def blend_channel(self, ch1, ch2):
        # calculate R L1 L2 in formula
        L1 = PyramidBlender.PyramidBuilder(
            ch1, self.levels_number).construct_laplacian_pyramid()
        L2 = PyramidBlender.PyramidBuilder(
            ch2, self.levels_number).construct_laplacian_pyramid()
        R = PyramidBlender.PyramidBuilder(
            self.mask.copy(), self.levels_number).construct_gaussian_pyramid()
        # construct blended pyramid
        blended_pyramid = PyramidBlender.construct_blended_pyramid(L1, L2, R, self.levels_number)
        # reconstruct channel based on step 4 in algorithm
        blended_channel = PyramidBlender.reconstruct_channel(blended_pyramid)
        return blended_channel

    def blend(self):
        result = np.zeros(self.image1.shape)
        # blend each channel
        for channel in range(3):
            result[:, :, channel] = self.blend_channel(
                self.image1[:, :, channel], self.image2[:, :, channel])

        return result


def direct_merge_of(source, target, mask):
    direct_merge = np.zeros_like(source)
    for i in range(3):
        direct_merge[:, :, i] = mask * source[:, :, i] + (1 - mask) * target[:, :, i]
    
    return direct_merge

experiment_number = 3

def main():
    source_path = f'sources/source{experiment_number}.jpg'
    target_path = f'targets/target{experiment_number}.jpg'
    blended_result_path = f'results_blended/res{experiment_number}.jpg'
    direct_merge_path = f'results_direct_merge/res{experiment_number}.jpg'
    mask_path = f'masks/mask{experiment_number}.jpg'
    
    levels_number = 6
    
    # read source and target image
    source = cv.imread(source_path)
    target = cv.imread(target_path)

    # initiate polygon maker (mouse click handler)
    first_window_name = 'Create polygon (press any key to continue)'
    polygon_maker = PolygonMaker.ClickHandler(source, first_window_name)
    cv.setMouseCallback(first_window_name, polygon_maker.click_event)
    cv.waitKey(0)
    cv.destroyWindow(first_window_name)
    print('polygon created')

    # initiate polygon mover (mouse move handler)
    filled_polygon = PolygonMaker(
        polygon_maker.get_points(), source.shape).get_filled_polygon()

    second_window_name = 'Move polygon to create mask (press c to continue)'
    delta, moved_polygon = PolygonMover(
        target, filled_polygon, second_window_name).get_moved_polygon()
    delta_x, delta_y = delta
    
    source_h, source_w, _ = source.shape
  
    T = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
    
    source_shifted = cv.warpAffine(source, T, (source_w, source_h))
    
  
    mask = (moved_polygon / 255).astype(np.float64)
    source_shifted = source_shifted.astype(np.float64)
    target = target.astype(np.float64)
    
    plt.imsave(mask_path, mask)
    
    cv.imwrite('s.jpg', source_shifted)

    blended = PyramidBlender(source_shifted,
                             target,
                             mask,
                             levels_number).blend().astype(np.uint8)
    
    
    direct_merge = direct_merge_of(source_shifted, target, mask) 

    cv.imwrite(direct_merge_path, direct_merge)    
    cv.imwrite(blended_result_path, blended)
    print('blended')


if __name__ == '__main__':
    main()

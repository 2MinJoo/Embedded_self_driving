from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
#import curses
import queue
import random

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
camera.awb_mode = 'fluorescent'
camera.exposure_mode = 'fixedfps'
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)

show_display = True

roi_bottom_x = 0
roi_bottom_y = 0
roi_top_x = 50
roi_top_y = 0
th_hist_pixel = 50000
th_hist_val = 100
theta_thresh = 25

dir_var_th = 400
dir_array_size = 5

pwm_rate_l = 0.73
pwm_rate_r = 1
dir_ratio = 70
dir_offset = 40

dir_val_prev = 0
dir_val_c = 0

pi = 3.14159265

buf_left = np.zeros((0), dtype=int, order='C')
buf_right = np.zeros((0), dtype=int, order='C')
buf_ = np.zeros((0), dtype=int, order='C')
buf_slo_left = 0
buf_slo_right = 0

print(buf_left.size)

fit_result, l_fit_result, r_fit_result, L_lane, R_lane, lane = [], [], [], [], [], []

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest_(img, vertices, color3=(255, 255, 255), color1=255):

    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        color = color3
    else:
        color = color1

    cv2.fillPoly(mask, vertices, color)

    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):

    # lines = lines.astype('uint8')

    # for line in lines:
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([], np.int32), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_arr = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    return lines

def grayscale_filter(img, factor=30, kernel_num=5):
    height, width = img.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    factor_img = np.zeros((height, width), np.uint8)

    factor_img = np.array(img, np.int16)

    factor_table = (abs(factor_img[:, :, 0] - factor_img[:, :, 1]) < factor) &\
                   (abs(factor_img[:, :, 1] - factor_img[:, :, 2]) < factor) &\
                   (abs(factor_img[:, :, 2] - factor_img[:, :, 0]) < factor)

    #factor_table = (abs(img[:, :, 0] - img[:, :, 1]))

    mask[factor_table] = 255
    #cv2.imshow("factor_table", factor_table)

    kernel = np.ones((kernel_num, kernel_num), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("mask", mask)

    gray_img = grayscale(img)
    cv2.imshow("gray_filter", gray_img)
    ret = cv2.bitwise_and(gray_img, mask)

    return ret

def morphology(img, kernel_num = 5, iteration = 3):
    ret = np.copy(img)
    kernel = np.ones((kernel_num, kernel_num), np.uint8)
    for i in range(iteration):
        ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, kernel)
    for i in range(iteration):
        ret = cv2.morphologyEx(ret, cv2.MORPH_OPEN, kernel)

    return ret


def mark_img(img, blue_threshold=220, green_threshold=220, red_threshold=200):  # 흰색 차선 찾기

    #  BGR 제한 값
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]

    if len(img.shape) > 2:
        # BGR 제한 값보다 작으면 검은색으로
        thresholds = (img[:, :, 0] < bgr_threshold[0]) \
                     & (img[:, :, 1] < bgr_threshold[1]) \
                     & (img[:, :, 2] < bgr_threshold[2])
        img[thresholds] = [0, 0, 0]
    else:
        #thresholds = img[:, :] < int(sum(bgr_threshold[:]) / 3)
        thresholds = img[:, :] < blue_threshold
        img[thresholds] = 0
    return img


def get_label(img):
    height, width = img.shape[:2]
    ret = np.zeros((height, width), np.uint8)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

    max_area = max(stats[1:, cv2.CC_STAT_AREA])
    max_area_label = 1
    for i in range(nlabels):
        if stats[i, cv2.CC_STAT_AREA] == max_area:
            max_area_label = i

    label_area = (labels[:, :] == max_area_label)
    ret[label_area] = 255

    return ret


def label_filter(img, labels, stats):

    ret = img

    return ret

def thresh_hist(img, nPixels=20000, th_min=200):
    hist = cv2.calcHist([img], [0], None, [256], [100, 256])
    sum =0
    for i, size in reversed(list(enumerate(hist))):
        sum += size
        if sum > nPixels:
            break
    th = i
    if th < th_min:
        th = th_min
    th_ret, th_img = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
    return th_img


def test_img(img):
    height, width = img.shape[:2]
    buf_l = buf_left.copy()
    buf_r = buf_right.copy()
    buf_slo_l = buf_slo_left
    buf_slo_r = buf_slo_right
    dir_val = dir_val_prev

    vertices = np.array(
        [[(roi_bottom_x, height - roi_bottom_y), (roi_top_x, roi_top_y),
          (width - roi_top_x, roi_top_y), (width - roi_bottom_x, height - roi_bottom_y)]],
        dtype=np.int32)
    ROI_img1 = region_of_interest_(img, vertices)
    # cv2.imshow("img", ROI_img1)

    #  흰색 차선 검출한 부분을 원본 image에 overlap 하기
    # color_thresholds = (mark[:, :, 0] == 0) & (mark[:, :, 1] == 0) & (mark[:, :, 2] > 200)
    # img[color_thresholds] = [0, 0, 255]

    blur_img = gaussian_blur(ROI_img1, 3)
    #cv2.imshow("blur", blur_img)

    gray_img = grayscale(blur_img)
    # gray_img = grayscale_filter(blur_img, 5)
    #cv2.imshow("gray", gray_img)

    equ_img = cv2.equalizeHist(gray_img)
    #cv2.imshow("equ", equ_img)

    # mark = np.copy(gray_img)  # roi_img 복사
    # mark = mark_img(mark, 220, 255, 255)
    # cv2.imshow("mark", mark)

    # th_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    th_img = thresh_hist(gray_img, th_hist_pixel, th_hist_val)

    morp_img = morphology(th_img, 5, 5)
    #cv2.imshow("morp", morp_img)

    # label_img = get_label(morp_img)
    # cv2.imshow("label", label_img)

    # nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(morp_img)

    canny_img = canny(morp_img, 60, 130)
    if show_display:
        cv2.imshow("canny", canny_img)

    # vertices = np.array(
    #    [[(0, height), (150, 100), (width - 150, 100), (width, height)]],
    #    dtype=np.int32)
    # ROI_img = region_of_interest_(canny_img, vertices, (0, 0, 255))

    # cv2.imshow("gray", ROI_img)

    line_arr = hough_lines(canny_img, 1, 1 * np.pi / 180, 20, 10, 30)

    # draw_lines(img, line_arr, thickness=2)

    line_arr = np.squeeze(line_arr)

    if line_arr.size < 2:
        print("none out")
        return img, buf_l, buf_r, buf_slo_l, buf_slo_r, dir_val

    if line_arr.size < 5:
        line_arr = line_arr.reshape((1, 4))

    # draw_lines(img, line_arr, [64, 64, 64], thickness=2)
    # print(line_arr.shape)
    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    # ignore horizontal slope lines
    line_arr = line_arr[np.abs(slope_degree) < 180 - theta_thresh]
    slope_degree = slope_degree[np.abs(slope_degree) < 180 - theta_thresh]
    line_arr = line_arr[np.abs(slope_degree) > theta_thresh]
    slope_degree = slope_degree[np.abs(slope_degree) > theta_thresh]

    draw_lines(img, line_arr, [192, 192, 192], thickness=2)

    # get mean x of lines
    mean_x = (sum(line_arr[:, 0]) + sum(line_arr[:, 2]))

    if len(line_arr) != 0:
        mean_x = mean_x / len(line_arr[:]) / 2
    else:
        print("none out")
        return img, buf_l, buf_r, buf_slo_l, buf_slo_r, dir_val

    line_arr_l = line_arr[line_arr[:, 0] < mean_x]
    line_arr_r = line_arr[line_arr[:, 0] >= mean_x]

    print(len(line_arr_l))

    if line_arr_l.size == 0 & line_arr_r.size == 0:
        print("none out")
        return img, buf_l, buf_r, buf_slo_l, buf_slo_r, dir_val

    if line_arr_l.size == 0:
        if buf_l.size == 0:
            return img, buf_l, buf_r, buf_slo_l, buf_slo_r, dir_val
        else:
            line_arr_l = buf_l.copy()
    else:
        buf_l = line_arr_l.copy()

    if line_arr_r.size == 0:
        if buf_r.size == 0:
            return img, buf_l, buf_r, buf_slo_l, buf_slo_r, dir_val
        else:
            line_arr_r = buf_r.copy()
    else:
        buf_r = line_arr_r.copy()

    mean_x_line = np.array([[int(mean_x), 0, int(mean_x), height]])
    draw_lines(img, mean_x_line, [128, 128, 128], thickness=2)

    # print(line_arr_l)
    # print(line_arr_r)

    slope_degree_l = (np.arctan2(line_arr_l[:, 1] - line_arr_l[:, 3],
                                 line_arr_l[:, 2] - line_arr_l[:, 0]) * 180) / np.pi
    slope_degree_r = (np.arctan2(line_arr_r[:, 1] - line_arr_r[:, 3],
                                 line_arr_r[:, 2] - line_arr_r[:, 0]) * 180) / np.pi

    slope_degree_l[:] = np.where(slope_degree_l < 0, slope_degree_l[:] + 180, slope_degree_l[:])
    slope_degree_r[:] = np.where(slope_degree_r < 0, slope_degree_r[:] + 180, slope_degree_r[:])

    draw_lines(img, line_arr_l, [0, 255, 0], thickness=2)
    draw_lines(img, line_arr_r, [255, 0, 0], thickness=2)

    degree_l = sum(abs(slope_degree_l[:]) / slope_degree_l.size)
    degree_l = degree_r = 0
    for x in slope_degree_l:
        degree_l = degree_l + abs(x) / slope_degree_l.size
    for x in slope_degree_r:
        degree_r = degree_r + abs(x) / slope_degree_r.size
    degree = (degree_l + degree_r) / 2

    if degree_l == 0:
        if buf_slo_l == 0:
            return img, buf_l, buf_r, buf_slo_l, buf_slo_r, dir_val
        else:
            degree_l = buf_slo_l
    else:
        buf_slo_l = degree_l

    if degree_r == 0:
        if buf_slo_r == 0:
            return img, buf_l, buf_r, buf_slo_l, buf_slo_r, dir_val
        else:
            degree_r = buf_slo_r
    else:
        buf_slo_r = degree_r

    center_l = [int(sum(line_arr_l[:, 0] + line_arr_l[:, 2]) / slope_degree_l.size / 2),
                int(sum(line_arr_l[:, 1] + line_arr_l[:, 3]) / slope_degree_l.size / 2)]
    center_r = [int(sum(line_arr_r[:, 0] + line_arr_r[:, 2]) / slope_degree_r.size / 2),
                int(sum(line_arr_r[:, 1] + line_arr_r[:, 3]) / slope_degree_r.size / 2)]

    cv2.circle(img, (center_l[0], center_l[1]), 3, [0, 255, 0], 5)
    cv2.circle(img, (center_r[0], center_r[1]), 3, [255, 0, 0], 5)

    degree_line_l = np.array([[int(center_l[0] - (height - center_l[1]) / np.tan(degree_l * pi / 180)), height,
                               int(center_l[0] + center_l[1] / np.tan(degree_l * pi / 180)), 0]])
    draw_lines(img, degree_line_l, [0, 255, 0], 5)

    degree_line_r = np.array([[int(center_r[0] - (height - center_r[1]) / np.tan(degree_r * pi / 180)), height,
                               int(center_r[0] + center_r[1] / np.tan(degree_r * pi / 180)), 0]])
    draw_lines(img, degree_line_r, [255, 0, 0], 5)

    # degree_line = [[int(width/2), height,
    #                int((width/2) + height/np.tan(degree*pi/180)), 0]]
    degree_line = np.array([[int(width / 2), height,
                             int((degree_line_l[0, 2] + degree_line_r[0, 2]) / 2), 0]])
    draw_lines(img, degree_line, [0, 0, 255], 5)

    dir_val = int((degree_line_l[0, 2] + degree_line_r[0, 2]) / 2) - int(width / 2)

    # print(line_arr.shape)
    # print(slope_degree)

    # interp = Collect_points(line_arr)

    # print(interp)

    return img, buf_l, buf_r, buf_slo_l, buf_slo_r, dir_val


acc = 0
dir = 0

left_pwm = 19
left_dir = 26
right_pwm = 6
right_dir = 13
tri_1 = 20
tri_2 = 21

auto = False
dir_array = np.zeros(dir_array_size)
dir_array.fill(dir_offset)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(left_pwm, GPIO.OUT)
GPIO.setup(left_dir, GPIO.OUT)
GPIO.setup(right_pwm, GPIO.OUT)
GPIO.setup(right_dir, GPIO.OUT)
GPIO.setup(tri_1, GPIO.OUT)
GPIO.setup(tri_2, GPIO.OUT)

GPIO.output(left_dir, False)
GPIO.output(right_dir, True)
GPIO.output(tri_1, True)
GPIO.output(tri_2, True)

left_p = GPIO.PWM(left_pwm, 100)
left_p.start(0)
right_p = GPIO.PWM(right_pwm, 100)
right_p.start(0)

#if not show_display:
#    # init the curses screen
#    stdscr = curses.initscr()
#    # use cbreak to not require a return key press
#    curses.cbreak()
#    curses.noecho()

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    #img = img[50:, :]
    img = frame.array

    result, buf_left, buf_right, buf_slo_left, buf_slo_right, dir_val_c = test_img(img)

    if show_display:
        cv2.imshow("result", result)

    key = cv2.waitKey(500) & 0xFF

    rawCapture.truncate(0)

#    if not show_display:
#        key = stdscr.getch()
    
    if key == ord("d"):
        dir = dir - 1
    if key == ord("a"):
        dir = dir + 1
    if key == ord("w"):
        acc = acc + 1
    if key == ord("s"):
        acc = acc - 1
    if key == ord("g"):
        auto = not auto

    if key == ord("r"):
        GPIO.output(tri_1, True)
        GPIO.output(tri_2, False)
        print("motor ccw")
    if key == ord("e"):
        GPIO.output(tri_1, False)
        GPIO.output(tri_2, True)
        print("motor cw")
    if key == ord("t"):
        GPIO.output(tri_1, True)
        GPIO.output(tri_2, True)
        print("motor off")

    # auto
    if auto:
        acc = 8
        dir_var = dir_val_c - dir_val_prev
        if dir_var < dir_var_th:
            dir_array = np.copy(dir_array[1:])
            dir_array = np.append(dir_array, dir_val_c)

        dir_pixel = sum(dir_array) / dir_array_size

        dir_val_prev = dir_val_c

        dir = -(dir_pixel - dir_offset) / dir_ratio

    if acc < 0:
        acc = 0
    if acc > 10:
        acc = 10

    if dir < -5:
        dir = -5
    if dir > 5:
        dir = 5

    left_pwm_v = (acc * 10) + (dir * -5)
    if left_pwm_v < 0:
        left_pwm_v = 0
        dir = 0
    if left_pwm_v > 100:
        left_pwm_v = 100

    right_pwm_v = (acc * 10) + (dir * 5)
    if right_pwm_v < 0:
        right_pwm_v = 0
        dir = 0
    if right_pwm_v > 100:
        right_pwm_v = 100

    left_p.ChangeDutyCycle(left_pwm_v * pwm_rate_l)
    right_p.ChangeDutyCycle(right_pwm_v * pwm_rate_r)

    if show_display:
        print('left : ', left_pwm_v)
        print('right : ', right_pwm_v)
        print('dir : ', dir)

    if key == ord("q"):
        left_p.ChangeDutyCycle(0)
        right_p.ChangeDutyCycle(0)
        break

#if not show_display:
#    curses.nocbreak()
#    stdscr.keypad(0)
#    curses.echo()
#    curses.endwin()

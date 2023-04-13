from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
import random

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)

roi_bottom_x = 50
roi_bottom_y = 50
roi_top_x = 100
roi_top_y = 100
theta_thresh = 45
pi = 3.14159265

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

def mark_img(img, blue_threshold=220, green_threshold=220, red_threshold=200):

    bgr_threshold = [blue_threshold, green_threshold, red_threshold]

    thresholds = (img[:, :, 0] < bgr_threshold[0]) \
                 & (img[:, :, 1] < bgr_threshold[1]) \
                 & (img[:, :, 2] < bgr_threshold[2])
    img[thresholds] = [0, 0, 0]
    return img

def test_img(img):
    height, width = img.shape[:2]

    vertices = np.array(
        [[(roi_bottom_x, height - roi_bottom_y), (roi_top_x, roi_top_y),
          (width - roi_top_x, roi_top_y), (width - roi_bottom_x, height - roi_bottom_y)]],
        dtype=np.int32)
    ROI_img1 = region_of_interest_(img, vertices, (0, 0, 255))

    mark = np.copy(ROI_img1)
    mark = mark_img(mark)

    #cv2.imshow("img", mark)
    color_thresholds = (mark[:, :, 0] == 0) & (mark[:, :, 1] == 0) & (mark[:, :, 2] > 200)
    #img[color_thresholds] = [0, 0, 255]

    blur_img = gaussian_blur(mark, 3)

    canny_img = canny(blur_img, 60, 130)

    vertices = np.array(
        [[(0, height), (150, 100), (width - 150, 100), (width, height)]],
        dtype=np.int32)
    ROI_img = region_of_interest_(canny_img, vertices, (0, 0, 255))

    cv2.imshow("gray", ROI_img)

    line_arr = hough_lines(ROI_img, 1, 1 * np.pi / 180, 10, 20, 30)

    # draw_lines(img, line_arr, thickness=2)

    line_arr = np.squeeze(line_arr)

    print(line_arr.size)

    if line_arr.size == 1 :
        print("none out")
        return img

    if line_arr.size < 5:
        line_arr = line_arr.reshape((1, 4))

    print(line_arr.shape)
    #print(line_arr)
    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    # ignore horizontal slope lines
    line_arr = line_arr[np.abs(slope_degree) < 180 - theta_thresh]
    slope_degree = slope_degree[np.abs(slope_degree) < 180 - theta_thresh]
    line_arr = line_arr[np.abs(slope_degree) > theta_thresh]
    slope_degree = slope_degree[np.abs(slope_degree) > theta_thresh]

    # get mean x of lines
    mean_x = (sum(line_arr[:, 0]) + sum(line_arr[:, 2]))

    if len(line_arr[:]) != 0:
        mean_x = mean_x / len(line_arr[:]) / 2
    else:
        mean_x = mean_x / 2

    line_arr_l = line_arr[line_arr[:, 0] < mean_x]
    line_arr_r = line_arr[line_arr[:, 0] >= mean_x]

    print(line_arr_l)
    print(line_arr_r)

    slope_degree_l = (np.arctan2(line_arr_l[:, 1] - line_arr_l[:, 3],
                                 line_arr_l[:, 2] - line_arr_l[:, 0]) * 180) / np.pi
    slope_degree_r = (np.arctan2(line_arr_r[:, 1] - line_arr_r[:, 3],
                                 line_arr_r[:, 2] - line_arr_r[:, 0]) * 180) / np.pi

    slope_degree_l[:] = np.where(slope_degree_l < 0, slope_degree_l[:] + 180, slope_degree_l[:])
    slope_degree_r[:] = np.where(slope_degree_r < 0, slope_degree_r[:] + 180, slope_degree_r[:])

    draw_lines(img, line_arr_l, [0, 255, 0], thickness=2)
    draw_lines(img, line_arr_r, [255, 0, 0], thickness=2)

    degree_l = degree_r = 0
    for x in slope_degree_l:
        degree_l = degree_l + abs(x) / slope_degree_l.size
    for x in slope_degree_r:
        degree_r = degree_r + abs(x) / slope_degree_r.size
    degree = (degree_l + degree_r) / 2
    #print(degree_l, degree_r, degree)
    #print(np.tan(degree*pi/180))

    if slope_degree_l.size == 0:
        devide_l = 1
        degree_l = 90
    else:
        devide_l = slope_degree_l.size

    if slope_degree_r.size == 0:
        devide_r = 1
        degree_r = 90
    else:
        devide_r = slope_degree_r.size

    center_l = [int(sum(line_arr_l[:, 0] + line_arr_l[:, 2]) / devide_l / 2),
                int(sum(line_arr_l[:, 1] + line_arr_l[:, 3]) / devide_l / 2)]
    center_r = [int(sum(line_arr_r[:, 0] + line_arr_r[:, 2]) / devide_r / 2),
                int(sum(line_arr_r[:, 1] + line_arr_r[:, 3]) / devide_r / 2)]

    cv2.circle(img, (center_l[0], center_l[1]), 3, [0, 255, 0], 5)
    cv2.circle(img, (center_r[0], center_r[1]), 3, [255, 0, 0], 5)

    degree_line_l = np.array([[int(center_l[0] - (height - center_l[1]) / np.tan(degree_l*pi/180)), height,
                               int(center_l[0] + center_l[1] / np.tan(degree_l*pi/180)), 0]])
    draw_lines(img, degree_line_l, [0, 255, 0], 5)

    degree_line_r = np.array([[int(center_r[0] - (height - center_r[1]) / np.tan(degree_r*pi/180)), height,
                               int(center_r[0] + center_r[1] / np.tan(degree_r*pi/180)), 0]])
    draw_lines(img, degree_line_r, [255, 0, 0], 5)

    # degree_line = [[int(width/2), height,
    #                int((width/2) + height/np.tan(degree*pi/180)), 0]]
    degree_line = np.array([[int(width/2), height,
                             int((degree_line_l[0, 2] + degree_line_r[0, 2]) / 2), 0]])
    draw_lines(img, degree_line, [0, 0, 255], 5)



    #print(line_arr.shape)
    #print(slope_degree)

    #interp = Collect_points(line_arr)

    #print(interp)

    return img

acc = 0
dir = 0

left_pwm = 19
left_dir = 26
right_pwm = 6
right_dir = 13
tri_1 = 20
tri_2 = 21

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

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):


    #img = img[50:, :]
    img = frame.array

    result = test_img(img)

    cv2.imshow("result", result)

    key = cv2.waitKey(100) & 0xFF

    rawCapture.truncate(0)
    
    if key == ord("d"):
        dir = dir - 1
    if key == ord("a"):
        dir = dir + 1
    if key == ord("w"):
        acc = acc + 1
    if key == ord("s"):
        acc = acc - 1

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

    left_p.ChangeDutyCycle(left_pwm_v)
    right_p.ChangeDutyCycle(right_pwm_v)

    print('left : ', left_pwm_v)
    print('right : ', right_pwm_v)

    if key == ord("q"):
        left_p.ChangeDutyCycle(0)
        right_p.ChangeDutyCycle(0)
        break

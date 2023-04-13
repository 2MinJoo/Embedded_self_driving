# from picamera.array import PiRGBArray
# from picamera import PiCamera
# import time
import cv2
import numpy as np
import random
import os, sys

# camera = PiCamera()
# camera.resolution = (640, 480)
# camera.framerate = 32
# rawCapture = PiRGBArray(camera, size=(640, 480))
# time.sleep(0.1)

# cap = cv2.VideoCapture('C:/Users/Gee/mj/0218/challenge_video.mp4')
# image = cv2.imread('C:/Users/Gee/mj/14.jpg')

# ---params---
roi_bottom_x = 0
roi_bottom_y = 0
roi_top_x = 50
roi_top_y = 50
theta_thresh = 45
pi = 3.14159265

buf_left = np.zeros((0), dtype=int, order='C')
buf_right = np.zeros((0), dtype=int, order='C')
buf_ = np.zeros((0), dtype=int, order='C')
print(buf_left.size)

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    #for line in lines:
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        #print(lines)


def draw_circle(img, lines, color=[0, 0, 255]):
    for line in lines:
        cv2.circle(img, (line[0], line[1]), 2, color, -1)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_arr = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_arr, lines)
    return lines


def weighted_img(img, initial_img):
    return cv2.addWeighted(initial_img, 0.8, img, 1, 0)


def Collect_points(lines):
    # reshape [:4] to [:2]
    interp = lines.reshape(lines.shape[0] * 2, 2)
    # interpolation & collecting points for RANSAC
    for line in lines:
        if np.abs(line[3] - line[1]) > 5:
            tmp = np.abs(line[3] - line[1])
            a = line[0];
            b = line[1];
            c = line[2];
            d = line[3]
            slope = (line[2] - line[0]) / (line[3] - line[1])
            for m in range(0, tmp, 5):
                if slope > 0:
                    new_point = np.array([[int(a + m * slope), int(b + m)]])
                    interp = np.concatenate((interp, new_point), axis=0)
                elif slope < 0:
                    new_point = np.array([[int(a - m * slope), int(b - m)]])
                    interp = np.concatenate((interp, new_point), axis=0)
    return interp


def get_random_samples(lines):
    one = random.choice(lines)
    two = random.choice(lines)
    if (two[0] == one[0]):  # extract again if values are overlapped
        while two[0] == one[0]:
            two = random.choice(lines)
    one, two = one.reshape(1, 2), two.reshape(1, 2)
    three = np.concatenate((one, two), axis=1)
    three = three.squeeze()
    return three


def compute_model_parameter(line):
    # y = mx+n
    m = (line[3] - line[1]) / (line[2] - line[0])
    n = line[1] - m * line[0]
    # ax+by+c = 0
    a, b, c = m, -1, n
    par = np.array([a, b, c])
    return par


def compute_distance(par, point):
    # distance between line & point

    return np.abs(par[0] * point[:, 0] + par[1] * point[:, 1] + par[2]) / np.sqrt(par[0] ** 2 + par[1] ** 2)


def model_verification(par, lines):
    # calculate distance
    distance = compute_distance(par, lines)
    # total sum of distance between random line and sample points
    sum_dist = distance.sum(axis=0)
    # average
    avg_dist = sum_dist / len(lines)

    return avg_dist


def draw_extrapolate_line(img, par, color=(0, 0, 255), thickness=2):
    x1, y1 = int(-par[1] / par[0] * img.shape[0] - par[2] / par[0]), int(img.shape[0])
    x2, y2 = int(-par[1] / par[0] * (img.shape[0] / 2 + 100) - par[2] / par[0]), int(img.shape[0] / 2 + 100)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def get_fitline(img, f_lines):
    rows, cols = img.shape[:2]
    output = cv2.fitLine(f_lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
    x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)
    result = [x1, y1, x2, y2]

    return result


def draw_fitline(img, result_l, result_r, color=(255, 0, 255), thickness=10):
    # draw fitting line
    lane = np.zeros_like(img)
    cv2.line(lane, (int(result_l[0]), int(result_l[1])), (int(result_l[2]), int(result_l[3])), color, thickness)
    cv2.line(lane, (int(result_r[0]), int(result_r[1])), (int(result_r[2]), int(result_r[3])), color, thickness)
    # add original image & extracted lane lines
    final = weighted_img(lane, img)
    return final


def draw_fitline_1(img, result_l, result_r, result, color=(255, 0, 255), thickness=10):
    # draw fitting line
    lane = np.zeros_like(img)
    cv2.line(lane, (int(result_l[0]), int(result_l[1])), (int(result_l[2]), int(result_l[3])), color, thickness)
    cv2.line(lane, (int(result_r[0]), int(result_r[1])), (int(result_r[2]), int(result_r[3])), color, thickness)
    cv2.line(lane, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), color, thickness)
    # add original image & extracted lane lines
    final = weighted_img(lane, img)
    return final


def erase_outliers(par, lines):
    # distance between best line and sample points
    distance = compute_distance(par, lines)

    # filtered_dist = distance[distance<15]
    filtered_lines = lines[distance < 13, :]
    return filtered_lines


def smoothing(lines, pre_frame):
    # collect frames & print average line
    lines = np.squeeze(lines)
    avg_line = np.array([0, 0, 0, 0])

    for ii, line in enumerate(reversed(lines)):
        if ii == pre_frame:
            break
        avg_line += line
    avg_line = avg_line / pre_frame

    return avg_line


def ransac_line_fitting(img, lines, min=100):
    global fit_result, l_fit_result, r_fit_result
    best_line = np.array([0, 0, 0])
    if (len(lines) != 0):
        for i in range(30):
            sample = get_random_samples(lines)
            parameter = compute_model_parameter(sample)
            cost = model_verification(parameter, lines)
            if cost < min:  # update best_line
                min = cost
                best_line = parameter
            if min < 3: break
        # erase outliers based on best line
        filtered_lines = erase_outliers(best_line, lines)
        fit_result = get_fitline(img, filtered_lines)
    else:
        if (fit_result[3] - fit_result[1]) / (fit_result[2] - fit_result[0]) < 0:
            l_fit_result = fit_result
            return l_fit_result
        else:
            r_fit_result = fit_result
            return r_fit_result

    if (fit_result[3] - fit_result[1]) / (fit_result[2] - fit_result[0]) < 0:
        l_fit_result = fit_result
        return l_fit_result
    else:
        r_fit_result = fit_result
        return r_fit_result

#
# def detect_lanes_img(img):
#     height, width = img.shape[:2]
#
#     # Set ROI
#     vertices = np.array(
#         [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
#         dtype=np.int32)
#     ROI_img = region_of_interest(img, vertices)
#
#     # Convert to grayimage
#     # g_img = grayscale(img)
#
#     # Apply gaussian filter
#     blur_img = gaussian_blur(ROI_img, 3)
#
#     # Apply Canny edge transform
#     canny_img = canny(blur_img, 70, 210)
#     # to except contours of ROI image
#     vertices2 = np.array(
#         [[(52, height), (width / 2 - 43, height / 2 + 62), (width / 2 + 43, height / 2 + 62), (width - 52, height)]],
#         dtype=np.int32)
#     canny_img = region_of_interest(canny_img, vertices2)
#
#     # Perform hough transform
#     # Get first candidates for real lane lines
#     line_arr = hough_lines(canny_img, 1, 1 * np.pi / 180, 30, 10, 20)
#
#     # draw_lines(img, line_arr, thickness=2)
#
#     line_arr = np.squeeze(line_arr)
#     # Get slope degree to separate 2 group (+ slope , - slope)
#     slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi
#
#     # ignore horizontal slope lines
#     line_arr = line_arr[np.abs(slope_degree) < 160]
#     slope_degree = slope_degree[np.abs(slope_degree) < 160]
#     # ignore vertical slope lines
#     line_arr = line_arr[np.abs(slope_degree) > 95]
#     slope_degree = slope_degree[np.abs(slope_degree) > 95]
#     L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
#     # print(line_arr.shape,'  ',L_lines.shape,'  ',R_lines.shape)
#
#     # interpolation & collecting points for RANSAC
#     L_interp = Collect_points(L_lines)
#     R_interp = Collect_points(R_lines)
#
#     # draw_circle(img,L_interp,(255,255,0))
#     # draw_circle(img,R_interp,(0,255,255))
#
#     # erase outliers based on best line
#     left_fit_line = ransac_line_fitting(img, L_interp)
#     right_fit_line = ransac_line_fitting(img, R_interp)
#
#     left = np.array(left_fit_line)
#     right = np.array(right_fit_line)
#
#     m_fit_line = (left+ right) / 2
#     m_fit_line = m_fit_line.astype('int32')
#
#     #print(m_fit_line)
#
#     # smoothing by using previous frames
#     L_lane.append(left_fit_line), R_lane.append(right_fit_line)
#     lane.append(m_fit_line)
#
#     if len(L_lane) > 10:
#         left_fit_line = smoothing(L_lane, 10)
#     if len(R_lane) > 10:
#         right_fit_line = smoothing(R_lane, 10)
#     if len(lane) > 10:
#         m_fit_line = smoothing(lane, 10)
#     #final = draw_fitline(img, left_fit_line, right_fit_line)
#     final = draw_fitline_1(img, left_fit_line, right_fit_line, m_fit_line)
#
#     return final

def region_of_interest_(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def test_img(img):
    height, width = img.shape[:2]
    buf_l = buf_left.copy()
    buf_r = buf_right.copy()

    vertices = np.array(
        [[(roi_bottom_x, height - roi_bottom_y), (roi_top_x, roi_top_y),
          (width - roi_top_x, roi_top_y), (width - roi_bottom_x, height - roi_bottom_y)]],
        dtype=np.int32)
    ROI_img1 = region_of_interest_(img, vertices)
    cv2.imshow("img", ROI_img1)
    mark = np.copy(ROI_img1)  # roi_img 복사
    mark = mark_img(mark)

    # cv2.imshow("img", mark)
  #  흰색 차선 검출한 부분을 원본 image에 overlap 하기
    color_thresholds = (mark[:, :, 0] == 0) & (mark[:, :, 1] == 0) & (mark[:, :, 2] > 200)
    #img[color_thresholds] = [0, 0, 255]



    blur_img = gaussian_blur(mark, 3)

    canny_img = canny(blur_img, 60, 130)

    vertices = np.array(
        [[(0, height), (150, 100), (width - 150, 100), (width, height)]],
        dtype=np.int32)
    ROI_img = region_of_interest_(canny_img, vertices, (0, 0, 255))

   # cv2.imshow("gray", ROI_img)

    line_arr = hough_lines(ROI_img, 1, 1 * np.pi / 180, 10, 20, 30)

    # draw_lines(img, line_arr, thickness=2)

    line_arr = np.squeeze(line_arr)

    if line_arr.size < 5 :
        line_arr = line_arr.reshape((1, 4))

    #print(line_arr.shape)
    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    # ignore horizontal slope lines
    line_arr = line_arr[np.abs(slope_degree) < 180 - theta_thresh]
    slope_degree = slope_degree[np.abs(slope_degree) < 180 - theta_thresh]
    line_arr = line_arr[np.abs(slope_degree) > theta_thresh]
    slope_degree = slope_degree[np.abs(slope_degree) > theta_thresh]

    # get mean x of lines
    mean_x = (sum(line_arr[:, 0]) + sum(line_arr[:, 2]) )

    if len(line_arr[:]) != 0:
        mean_x = mean_x / len(line_arr[:]) / 2
    else:
        print("none out")
        return img

    line_arr_l = line_arr[line_arr[:, 0] < mean_x]
    line_arr_r = line_arr[line_arr[:, 0] >= mean_x]

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

    degree_l = degree_r = 0
    for x in slope_degree_l:
        degree_l = degree_l + abs(x) / slope_degree_l.size
    for x in slope_degree_r:
        degree_r = degree_r + abs(x) / slope_degree_r.size
    degree = (degree_l + degree_r) / 2

    if slope_degree_l.size == 0:
        if buf_l.size == 0:
            return img
        else:
            slope_degree_l = buf_l.copy()
    else:
        buf_l = slope_degree_l.copy()

    if slope_degree_r.size == 0:
        if buf_r.size == 0:
            return img
        else:
            slope_degree_r = buf_r.copy()
    else:
        buf_r = slope_degree_r.copy()

    center_l = [int(sum(line_arr_l[:, 0] + line_arr_l[:, 2]) / slope_degree_l.size / 2),
                int(sum(line_arr_l[:, 1] + line_arr_l[:, 3]) / slope_degree_l.size / 2)]
    center_r = [int(sum(line_arr_r[:, 0] + line_arr_r[:, 2]) / slope_degree_r.size / 2),
                int(sum(line_arr_r[:, 1] + line_arr_r[:, 3]) / slope_degree_r.size / 2)]

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

    return img, buf_l, buf_r

def mark_img(img, blue_threshold=220, green_threshold=220, red_threshold=200):  # 흰색 차선 찾기

    #  BGR 제한 값
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]

    # BGR 제한 값보다 작으면 검은색으로
    thresholds = (img[:, :, 0] < bgr_threshold[0]) \
                 & (img[:, :, 1] < bgr_threshold[1]) \
                 & (img[:, :, 2] < bgr_threshold[2])
    img[thresholds] = [0, 0, 0]
    return img

# while (cap.isOpened()):
#     ret, frame = cap.read()
#
#     result = detect_lanes_img(frame)
#
#     cv2.imshow('result', result)
#
#     # out.write(frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()

while(1):
    for i in range(22):
        i = i + 1
       # i = 15
        s = str(i)
        if i < 10:
            s = '0' + s

        img = cv2.imread('Capture\\' + s + '.jpg', cv2.IMREAD_COLOR)
        #height, width = img.shape[:2]

        result, buf_left, buf_right = test_img(img)

        cv2.imshow("Frame", result)
        #cv2.imshow("camera", img)

        key = cv2.waitKey(300) & 0xFF

        if key == ord("q"):
            exit()

cv2.waitKey(0)

#result = test_img(image)

#cv2.imshow('result', result)
#cv2.waitKey(0)

import time
import cv2
import numpy as np
import random

fit_result, l_fit_result, r_fit_result, L_lane, R_lane, lane = [], [], [], [], [], []
line_arr2 = []

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([], np.int32), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_arr = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    return lines


def weighted_img(img, initial_img):
    return cv2.addWeighted(initial_img, 0.8, img, 1, 0)


def Collect_points(lines):
    # reshape [:4] to [:2]

    if lines.ndim > 1:
        interp = lines.reshape(lines.shape[0] * 2, 2)
    else:
        interp = lines.reshape(2, 2)
    # interpolation & collecting points for RANSAC
    for line in lines:
        if np.abs(line[3] - line[1]) > 5:
            tmp = np.abs(line[3] - line[1])
            a = line[0]
            b = line[1]
            c = line[2]
            d = line[3]
            slope = (line[2] - line[0]) / (line[3] - line[1])
            for m in range(0, tmp, 5):
                if slope > 0:
                    new_point = np.array([[int(a + m * slope), int(b + m)]], np.int32)
                    interp = np.concatenate((interp, new_point), axis=0)
                elif slope < 0:
                    new_point = np.array([[int(a - m * slope), int(b - m)]], np.int32)
                    interp = np.concatenate((interp, new_point), axis=0)
    return interp


def get_random_samples(lines):
    if not lines.size:
        return [0, 0, 0]

    if lines.size:
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
    if line[2] != line[0]:
        m = (line[3] - line[1]) / (line[2] - line[0])
        n = line[1] - m * line[0]
        # ax+by+c = 0
        a, b, c = m, -1, n
    else:
        a, b, c = 1, 0, -line[0]

    par = np.array([a, b, c], np.int32)
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


def get_fitline(img, f_lines):
    rows, cols = img.shape[:2]
    output = cv2.fitLine(f_lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]

    if vy != 0:
        x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
        #x2, y2 = int(((img.shape[0] / 2 - 100) - y) / vy * vx + x), int(img.shape[0] / 2 - 100)
        x2, y2 = int(((480 / 3) - y) / vy * vx + x), int(480 / 3)
        result = [x1, y1, x2, y2]
    else:
        result = [0, 0, 0, 0]

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
    avg_line = np.array([0, 0, 0, 0], dtype=np.int32)

    for ii, line in enumerate(reversed(lines)):
        if ii == pre_frame:
            break
        avg_line += line
    avg_line = avg_line / pre_frame

    return avg_line

def ransac_line_fitting_(img, lines, min=100):
    global fit_result, l_fit_result, r_fit_result
    best_line = np.array([0, 0, 0], np.int32)
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
    #print(filtered_lines)
    if filtered_lines.size:
        fit_result = get_fitline(img, filtered_lines)
    else:
        fit_result = [0, 0, 0, 0]

    return fit_result

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


def test_image(img):
    height, width = img.shape[:2]

    b_image = gaussian_blur(img, 3)
    c_image = canny(b_image, 80, 180)
    final = c_image

    #cv2.imshow("roi", c_image)

    line = hough_lines(c_image, 1, 1 * np.pi / 180, 30, 40, 30)
    line = np.squeeze(line)

    if line.ndim > 1:
        slope_degree = (np.arctan2((line[:, 1] - line[:, 3]), (line[:, 0] - line[:, 2]))) * 180 / np.pi

        line = line[np.abs(slope_degree) < 170]
        slope_degree = slope_degree[np.abs(slope_degree) < 170]
        # ignore vertical slope lines
        line = line[np.abs(slope_degree) > 95]
        slope_degree = slope_degree[np.abs(slope_degree) > 95]
        L_lines, R_lines = line[(slope_degree > 0), :], line[(slope_degree < 0), :]

        # interpolation & collecting points for RANSAC
        L_interp = Collect_points(L_lines)
        R_interp = Collect_points(R_lines)

        left_fit_line = ransac_line_fitting_(img, L_interp)
        right_fit_line = ransac_line_fitting_(img, R_interp)

        left = np.array(left_fit_line)
        right = np.array(right_fit_line)

        m_fit_line = (left + right) / 2
        m_fit_line = m_fit_line.astype('int32')

        print(m_fit_line)

        # smoothing by using previous frames
        L_lane.append(left_fit_line), R_lane.append(right_fit_line)
        lane.append(m_fit_line)

        if len(L_lane) > 10:
            left_fit_line = smoothing(L_lane, 10)
        if len(R_lane) > 10:
            right_fit_line = smoothing(R_lane, 10)
        if len(lane) > 10:
            m_fit_line = smoothing(lane, 10)
        final = draw_fitline_1(img, left_fit_line, right_fit_line, m_fit_line)
    else:
        final = img

    return final


# for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
#
#     img = frame.array
#
#     result = test_image(img)
#
#     cv2.imshow("Frame", result)
#     cv2.imshow("camera", img)
#
#     key = cv2.waitKey(30) & 0xFF
#
#     rawCapture.truncate(0)
#
#     if key == ord("q"):
#         break

cap = cv2.VideoCapture('C:/Users/Gee/mj/0218/project_video.mp4')

while (cap.isOpened()):
    ret, frame = cap.read()

    result = test_image(frame)

    cv2.imshow('result', result)

    # out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
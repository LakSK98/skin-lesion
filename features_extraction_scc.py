import cv2
import numpy as np
import math
from scipy.spatial import distance
from math import pi

def dot_segment_detection(original_image):
    gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # Sharpen the image using a sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen_img = cv2.filter2D(gray_img, -1, kernel)
    # Binary thresholding
    _, binary_img = cv2.threshold(sharpen_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = sharpen_img
    img = binary_img
    # Dilate the image to try to get circular shapes completed
    kernel = np.ones((3,3), np.uint8)
    gray = cv2.dilate(img, kernel, iterations=2)
    gray = cv2.erode(gray, kernel, iterations=3)
    # Set up SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 8
    params.maxArea = 5000  # This value obtained by observation with the dotted and glomerular vessels
    params.filterByColor = True
    params.blobColor = 0
    params.filterByCircularity = True
    params.minCircularity = 0.4  # Minimum value to be a circular shape
    params.maxCircularity = 1
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minDistBetweenBlobs = 8
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs
    keypoints = detector.detect(gray)
    # Create a mask for the detected keypoints
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros((height, width), np.uint8)
    dot_test_array = []
    new_list = []
    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        s = keypoint.size
        r = int(math.floor(s / 2))
        new_list.append([x, y, r])
        cv2.circle(mask, (x, y), radius=r, color=(255, 255, 255), thickness=-1)
        cv2.rectangle(mask, (x-r, y-r), (x+r, y+r), color=(255, 255, 255), thickness=-1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite("Test_vessel/blob_detected_before_color_check_original.jpg", blobs)
    for box in new_list:
        x = box[0]
        y = box[1]
        r = box[2]
        crp = original_image[y-r:y+r, x-r:x+r]
        dsize = (20, 20)
        crp = cv2.resize(crp, dsize)
        dot_test_array.append(crp)
    return masked_image, dot_test_array, 1 if len(dot_test_array) else 0

def get_threshold(small_img):
    bins_num = 256
    hist, bin_edges = np.histogram(small_img, bins=bins_num)
    hist = np.divide(hist.ravel(), hist.max())
    bin_mids = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_mids) / weight1
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    index_of_max_val = np.argmax(inter_class_variance)
    threshold = bin_mids[:-1][index_of_max_val]
    return threshold

def image_colorfulness(image):
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    rgMean, rgStd = np.mean(rg), np.std(rg)
    ybMean, ybStd = np.mean(yb), np.std(yb)
    stdRoot = np.sqrt((rgStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rgMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot), np.mean(R), np.mean(G), np.mean(B)

def predict_dotted_vessel_or_not(dotted_vessel_test_array):
    true_count = 0
    count = 0
    COLOR_VAL = 6  # minimum value to get the different color to be visualized between 2 colorfulness values
    RED_VAL = 5    # minimum value to visualize the redness of the feature from its background
    BLACK_VAL = 4  # minimum value to remove black noises of the feature from its background
    MIN_COUNT = 3  # minimum no of vessels should presence in the image to consider it as vessel feature
    MIN_PROB = 0.8 # atleast 0.1 probability should there for detect as vessels feature
    colorful_feature = []
    colorful_background = []
    mean_red_value = []
    mean_blue_value = []
    for file in dotted_vessel_test_array:
        img = file
        small_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold = get_threshold(small_img)
        ret, thresh1 = cv2.threshold(small_img, threshold, 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(small_img, threshold, 255, cv2.THRESH_BINARY_INV)

        feature = cv2.bitwise_and(img, img, mask=thresh1)
        background = cv2.bitwise_and(img, img, mask=thresh2)

        f_gray = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        count1 = cv2.countNonZero(f_gray)
        count2 = cv2.countNonZero(b_gray)

        if count1 > BLACK_VAL and count2 > BLACK_VAL: #remove black values - mostly they are noises
          I, R, G, B = image_colorfulness(feature)
          I1, R1, G1, B1 = image_colorfulness(background)
          if (I - COLOR_VAL < I1): #colorfulness of the feature should greater than of background.
              count = count + 1
          if (R > G and R > B and R > (R1 + RED_VAL)): #redness of the feature should greater than that of background.
              true_count = true_count + 1
              colorful_feature.append(I)
              colorful_background.append(I1)
              mean_red_value.append(R)
              mean_blue_value.append(B)
    if(np.mean(colorful_feature) > 45 and np.mean(colorful_background) > 45 and np.mean(mean_red_value) > 100 and np.mean(mean_blue_value) > 85):
        return 1
    else:
        return 0

def circle_segment_detection(eropen, sharpen_image, circle_binarize_image):
    VAL_THRESH = 252  # minimum value to be segment the binarized image
    MIN_AREA = 4  # minimum area contour to reduce the affect of the noises
    MAX_AREA = 1500  # maximum area contour
    MIN_APPROX = 5  # minimum line approximation to be a circle as of this has more noises
    MIN_CIR = 0.4  # minimum circularity of follicles are considered as 0.4 because with diameter of the distance can vary from the average distance
    DIST_CONST = 0.20  # Range of the distance can vary from the average distance
    MIN_COUNT_CON = 0.5  # minimum number of contours should have in the range defined to be a circle

    im = circle_binarize_image
    image = eropen
    im1 = image.copy()
    ret, im = cv2.threshold(im, VAL_THRESH, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    im = cv2.dilate(im, kernel, iterations=3)  # As the try of circular to be completed
    im = cv2.erode(im, kernel, iterations=3)
    imgray = im
    contours, hierarchy = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_area = []  # calculate area and filter into new array
    countor_mask = im1

    for con in contours:
        area = cv2.contourArea(con)
        if MIN_AREA < area < MAX_AREA:
            contours_area.append(con)

    contours_cirles = []
    circle_test_array = []
    new_list = []
    for con in contours_area:  # check if contour is of circular shape
        distance_list = []
        perimeter = cv2.arcLength(con, True)
        approx = cv2.approxPolyDP(con, 0.01 * cv2.arcLength(con, True), True)
        area = cv2.contourArea(con)
        circularity = 4 * math.pi * (area / (perimeter * perimeter))
        radius = (2 * area) / perimeter
        M = cv2.moments(con)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        center = (cx, cy)
        rect = []
        if perimeter == 0:
            break
        else:
            if (len(approx) > MIN_APPROX) and (circularity > MIN_CIR):
                # get extreme points of the contour
                top = tuple(con[con[:, :, 1].argmin()][0])
                left = tuple(con[con[:, :, 0].argmin()][0])
                right = tuple(con[con[:, :, 0].argmax()][0])
                bottom = tuple(con[con[:, :, 1].argmax()][0])
                for i in con:
                    [x, y] = i
                    distance_list.append(distance.euclidean((x, y), center))  # get coordinates

        d = len(distance_list)
        avg_distance = (sum(distance_list) / d)  # get the average contour
        intercept1 = avg_distance - (avg_distance * DIST_CONST)  # lower level distance range
        intercept2 = avg_distance + (avg_distance * DIST_CONST)  # upper level distance range
        count = 0
        for j in distance_list:
            if (intercept1 <= j <= intercept2):
                count += 1  # count the no of contours inside the defined range of distances
        if (count / d * MIN_COUNT_CON):
            param1 = left[0]
            param2 = top[1]
            param3 = right[0]
            param4 = bottom[1]
            rect.append((param1, param2))
            rect.append((param3, param2))
            rect.append((param3, param4))
            rect.append((param1, param4))
            gg = np.array(rect)
            x, y, w, h = cv2.boundingRect(gg)
            rect = cv2.minAreaRect(gg)
            box = cv2.boxPoints(rect)
            box = np.int8(box)
            new_list.append(gg)
            countor_mask.append(con)
            contours_cirles.append(box)
            # this is for contour based mask
            # contours_area.append(box)
        # Create masks for detected circles
        mask = np.zeros_like(imgray)
        for contour in contours_cirles:
            cv2.drawContours(mask, [contour], 0, (255), thickness=cv2.FILLED)
        # Apply the mask to the original image to extract the white follicles
        result = cv2.bitwise_and(image, image, mask=mask)
        return result, contours_cirles

def circle_segment_detection(img):
    eropen = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    sharpen_image = cv2.GaussianBlur(img, (0,0), 3)
    sharpen_image = cv2.addWeighted(img, 1.5, sharpen_image, -0.5, 0)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, circle_binarize_image = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
    VAL_THRESH = 252  # minimum value to be segment the binarized image
    MIN_AREA = 4  # minimum area contour to reduce the affect of the noises
    MAX_AREA = 1500  # maximum area contour
    MIN_APPROX = 5  # minimum line approximation to be a circle as of this has more noises
    MIN_CIR = 0.4  # minimum circularity of follicles are considered as 0.4 because with diameter of the distance can vary from the average distance
    DIST_CONST = 0.20  # Range of the distance can vary from the average distance
    MIN_COUNT_CON = 0.5  # minimum number of contours should have in the range defined to be a circle
    im = circle_binarize_image
    image = eropen
    im1 = image.copy()
    ret, im = cv2.threshold(im, VAL_THRESH, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    im = cv2.dilate(im, kernel, iterations=3)  # As the try of circular to be completed
    im = cv2.erode(im, kernel, iterations=3)
    imgray = im
    contours, hierarchy = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_area = []  # calculate area and filter into new array
    countor_mask = im1
    for con in contours:
        area = cv2.contourArea(con)
        if MIN_AREA < area < MAX_AREA:
            contours_area.append(con)
    contours_cirles = []
    circle_test_array = []
    new_list = []
    for con in contours_area:  # check if contour is of circular shape
        distance_list = []
        perimeter = cv2.arcLength(con, True)
        approx = cv2.approxPolyDP(con, 0.01 * cv2.arcLength(con, True), True)
        area = cv2.contourArea(con)
        circularity = 4 * math.pi * (area / (perimeter * perimeter))
        radius = (2 * area) / perimeter
        M = cv2.moments(con)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        center = (cx, cy)
        rect = []
        if perimeter == 0:
            break
        else:
            if (len(approx) > MIN_APPROX) and (circularity > MIN_CIR):
                # get extreme points of the contour
                top = tuple(con[con[:, :, 1].argmin()][0])
                left = tuple(con[con[:, :, 0].argmin()][0])
                right = tuple(con[con[:, :, 0].argmax()][0])
                bottom = tuple(con[con[:, :, 1].argmax()][0])
                for i in con:
                    [x, y] = i
                    distance_list.append(distance.euclidean((x, y), center))  # get coordinates
        d = len(distance_list)
        avg_distance = (sum(distance_list) / d)  # get the average contour
        intercept1 = avg_distance - (avg_distance * DIST_CONST)  # lower level distance range
        intercept2 = avg_distance + (avg_distance * DIST_CONST)  # upper level distance range
        count = 0
        for j in distance_list:
            if (intercept1 <= j <= intercept2):
                count += 1  # count the no of contours inside the defined range of distances
        if (count / d * MIN_COUNT_CON):
            param1 = left[0]
            param2 = top[1]
            param3 = right[0]
            param4 = bottom[1]
            rect.append((param1, param2))
            rect.append((param3, param2))
            rect.append((param3, param4))
            rect.append((param1, param4))
            gg = np.array(rect)
            x, y, w, h = cv2.boundingRect(gg)
            rect = cv2.minAreaRect(gg)
            box = cv2.boxPoints(rect)
            box = np.int8(box)
            new_list.append(gg)
            countor_mask.append(con)
            contours_cirles.append(box)
            # this is for contour based mask
            # contours_area.append(box)
        # Create masks for detected circles
        mask = np.zeros_like(imgray)
        for contour in contours_cirles:
            cv2.drawContours(mask, [contour], 0, (255), thickness=cv2.FILLED)
        # Apply the mask to the original image to extract the white follicles
        result = cv2.bitwise_and(image, image, mask=mask)
        return result, contours_cirles
    
def hu_moment(point_array):  # get the center of mass of the given contours
    M = cv2.moments(point_array)
    if M['m00'] != 0:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
        return x, y
    return None, None

def angle_between_2_point(cx, cy, x, y, dis_1, dis_2, ang):  # angle calculation between 2 points
    if dis_1 == 0:
        if dis_2 == 0:
            angle = 0 + ang
        else:
            angle = 0 + ang if y > cy else 180 + ang
    elif dis_2 == 0:
        angle = 90 + ang if x < cx else 270 + ang
    else:
        angle = math.atan(dis_2 / dis_1) / pi * 180
        if ((y < cy) and angle < 0) or (not (y < cy) and angle > 0):
            angle = angle + 270 + ang
        else:
            angle = angle + 90 + ang
    return angle

def count_points(a, b, c, d, e, f, g, h, set1, set2, set3, set4, set5, set6, set7, set8, ang, angle, i):  # get the no of points
    if 0 <= angle - ang < 45:
        a = a + 1
        set1.append(i)
    elif 45 <= angle - ang < 90:
        b = b + 1
        set2.append(i)
    elif 90 <= angle - ang < 135:
        c = c + 1
        set3.append(i)
    elif 135 <= angle - ang < 180:
        d = d + 1
        set4.append(i)
    elif 180 <= angle - ang < 225:
        e = e + 1
        set5.append(i)
    elif 225 <= angle - ang < 270:
        f = f + 1
        set6.append(i)
    elif 270 <= angle - ang < 315:
        g = g + 1
        set7.append(i)
    else:
        h = h + 1
        set8.append(i)
    return a, b, c, d, e, f, g, h, set1, set2, set3, set4, set5, set6, set7, set8

def moment_check(center_list, set1):  # check the center of mass of each contour array in 8 different phases
    M = cv2.moments(np.array(set1))
    if M['m00'] != 0:
        center = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        center_list.append(center)
    return center_list

def rosette_segment_detection(img):
    sharpen_image = cv2.GaussianBlur(img, (0, 0), 3)
    sharpen_image = cv2.addWeighted(img, 1.5, sharpen_image, -0.5, 0)
    # _, binarized_image = cv2.threshold(sharpened_image, 127, 255, cv2.THRESH_BINARY)
    rosette_binarize_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    VAL_THRESH = 127  # minimum value to be segment the binarized image
    MIN_AREA = 20  # minimum area contour
    MAX_AREA = 1000  # maximum area contour
    MIN_APPROX = 8  # minimum line approximation to be a rosette
    im = sharpen_image
    img_gray = rosette_binarize_image
    ret, thresh = cv2.threshold(img_gray, VAL_THRESH, 255, 0)
    kernel = np.ones((3, 3), np.uint8)
    img_gray = cv2.dilate(thresh, kernel, iterations=1)
    rosette_test_array = []
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_area = []
    contour_mask = []
    new_list = []
    for con in contours:
        area = cv2.contourArea(con)
        if MIN_AREA <= area <= MAX_AREA:
            approx = cv2.approxPolyDP(con, 0.01 * cv2.arcLength(con, True), True)
            if len(approx) >= MIN_APPROX:
                cx, cy = hu_moment(con)
                center = (cx, cy)
                ang = 0
                center_list = []
                cumulative_centers = []
                a, b, c, d, e, f, g, h = 0, 0, 0, 0, 0, 0, 0, 0
                set1, set2, set3, set4, set5, set6, set7, set8 = [], [], [], [], [], [], [], []
                top = tuple(con[con[:, :, 1].argmin()][0])
                left = tuple(con[con[:, :, 0].argmin()][0])
                right = tuple(con[con[:, :, 0].argmax()][0])
                bottom = tuple(con[con[:, :, 1].argmax()][0])
                ang = angle_between_2_point(cx, cy, top[0], top[1], top[0], top[1], ang)
                for i in con:
                    [x, y] = i[0]
                    angle = angle_between_2_point(cx, cy, x, y, cx - x, cy - y, ang)
                    a, b, c, d, e, f, g, h, set1, set2, set3, set4, set5, set6, set7, set8 = count_points(
                        a, b, c, d, e, f, g, h, set1, set2, set3, set4, set5, set6, set7, set8, ang, angle, i
                    )
                center_list = moment_check(center_list, set1)
                center_list = moment_check(center_list, set2)
                center_list = moment_check(center_list, set3)
                center_list = moment_check(center_list, set4)
                center_list = moment_check(center_list, set5)
                center_list = moment_check(center_list, set6)
                center_list = moment_check(center_list, set7)
                center_list = moment_check(center_list, set8)
                rect = []
                if 0 <= len(center_list) <= 8:
                    if len(set1) > 2 and len(set2) > 2 and len(set3) > 2 and len(set4) > 2 and len(set5) > 2 and len(set6) > 2 and len(set7) > 2 and len(set8) > 2:
                        param1 = left[0] - 2
                        param2 = top[1] - 2
                        param3 = right[0] + 2
                        param4 = bottom[1] + 2
                        rect.append((param1, param2))
                        rect.append((param3, param2))
                        rect.append((param3, param4))
                        rect.append((param1, param4))
                        gg = np.array(rect)
                        x, y, w, h = cv2.boundingRect(gg)
                        rect = cv2.minAreaRect(gg)
                        box = cv2.boxPoints(rect)
                        box = np.int8(box)
                        new_list.append(gg)
                        contour_mask.append(con)  # con - this is for contour based mask
                        contours_area.append(box)
    return 1 if len(contours_area)> 0 else 0

def scc_extract_features(img):
    masked_image, dot_test_array, dotted_vessels_presence = dot_segment_detection(img)
    color_presence = predict_dotted_vessel_or_not(dot_test_array)
    white_follicles_presence = 0
    try:
        result, contours_cirles = circle_segment_detection(img)
        white_follicles_presence = 1
    except:
        white_follicles_presence = 0
    rosette_presence = rosette_segment_detection(img)
    return (1 if dotted_vessels_presence and color_presence else 0), white_follicles_presence, rosette_presence
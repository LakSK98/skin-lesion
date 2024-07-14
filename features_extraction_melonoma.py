import cv2
import numpy as np

# Function to remove hair from the image
def remove_hair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray_filtered, cv2.MORPH_BLACKHAT, kernel)
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(threshold, kernel, iterations=1)
    smooth = cv2.GaussianBlur(dilated, (5, 5), 0)
    result_telea = cv2.inpaint(image, smooth, 1, cv2.INPAINT_TELEA)
    result_ns = cv2.inpaint(image, smooth, 1, cv2.INPAINT_NS)
    result = cv2.addWeighted(result_telea, 0.5, result_ns, 0.5, 0)
    return result

# Function to extract blood vessels using thresholding and morphological operations
def segmented_img(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph

# Function to analyze color features
def color_analysis(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([20, 255, 255])
    mask_brown = cv2.inRange(hsv_image, lower_brown, upper_brown)
    mask = cv2.bitwise_or(mask_red, mask_brown)
    return mask

# Function to analyze texture features
def texture_analysis(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    abs_laplacian = np.absolute(laplacian)
    texture = np.uint8(abs_laplacian)
    return texture

# Function to analyze border irregularity
def border_analysis(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# Function to analyze size of the lesion
def size_analysis(binary_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    largest_component = max(stats[1:], key=lambda x: x[cv2.CC_STAT_AREA])
    area = largest_component[cv2.CC_STAT_AREA]
    return area

# Asymmetry Detection
def asymmetry_detection(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        roi = binary_image[y:y+h, x:x+w]
        half_width = w // 2
        left_half = roi[:, :half_width]
        right_half = roi[:, w-half_width:]
        right_half_flipped = cv2.flip(right_half, 1)
        asymmetry = cv2.absdiff(left_half, right_half_flipped)
        asymmetry_score = np.sum(asymmetry) / 255
        if asymmetry_score > 1:
            return 1
        else:
            return 0
    return 0

# Blue-White Veil Detection
def blue_white_veil_detection(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue_white = np.array([90, 0, 180])
    upper_blue_white = np.array([255, 30, 255])
    mask = cv2.inRange(hsv_image, lower_blue_white, upper_blue_white)
    score = np.sum(mask) / 255
    if score > 8000:
        return 1
    else:
        return 0

# Regression Structure Detection
def regression_structure_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    regression_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    score = np.sum(regression_mask) / 255
    if 168000 < score < 750000:
        return 1
    else:
        return 0

def melonoma_extract_features(image):
    # Remove hair from the image
    hair_removed_image = remove_hair(image)
    # Extract blood vessels
    segmented_image = segmented_img(hair_removed_image)
    # Asymmetry Detection
    asymmetry_presence = asymmetry_detection(segmented_image)
    # Blue-White Veil Detection
    blue_white_veil_presence = blue_white_veil_detection(hair_removed_image)
    # Regression Structure Detection
    regression_presence = regression_structure_detection(hair_removed_image)
    return asymmetry_presence, blue_white_veil_presence, regression_presence
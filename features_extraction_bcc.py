import cv2
import numpy as np

def extract_bw(image):
    # Convert image to green channel and enhance contrast
    b, green_channel, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(green_channel)
    # Sequential morphological operations to highlight features
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    opened = cv2.morphologyEx(contrast_enhanced, cv2.MORPH_OPEN, kernel_small)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_small)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_medium)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_large)
    subtracted = cv2.subtract(opened, contrast_enhanced)
    enhanced = clahe.apply(subtracted)
    # Thresholding to create a binary image
    _, binary = cv2.threshold(enhanced, 15, 255, cv2.THRESH_BINARY)
    mask = np.ones(enhanced.shape, dtype="uint8") * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    result = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    _, final = cv2.threshold(result, 15, 255, cv2.THRESH_BINARY_INV)
    final = cv2.erode(final, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
    final = cv2.erode(final, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)), iterations=1)
    # Calculate area of red vessels
    vessel_area = np.sum(final == 0)  # Count white pixels
    vessel_presence = int(vessel_area > 100)  # 1 if vessels are present, 0 otherwise
    # Highlight the vessels in red on a black background
    vessel_highlight = np.zeros_like(image)
    vessel_highlight[:,:,0] = 255  # Set red background
    vessel_highlight[final == 255, 0] = 0  # Mask vessels in black
    return final, vessel_highlight, vessel_area, vessel_presence

def extract_blue_gray_ovoids(image):
    # Convert image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range for blue-gray colors in HSV
    lower_bound = np.array([100, 50, 20])  # Lower HSV boundary
    upper_bound = np.array([140, 255, 120])  # Upper HSV boundary
    # Threshold the image to get only blue-gray areas
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(cnt) for cnt in contours)
    # Calculate presence based on the area
    presence = 1 if total_area > 261 else 0
    # Optional: Use the mask to extract the blue-gray areas from the original image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask, total_area, presence

def extract_ulceration_by_color(image):
    # Convert image to HSV for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the color range for red areas
    lower_red1 = np.array([0, 160, 50])   # Lower range for red
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 160, 50]) # Higher range for red
    upper_red2 = np.array([180, 255, 255])
    # Create masks to detect red areas
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    full_mask = cv2.add(mask1, mask2)
    # Apply the mask to extract red regions
    result = cv2.bitwise_and(image, image, mask=full_mask)
    return result, full_mask

def calculate_area_and_presence(mask, threshold=500):
    # Calculate the area as the number of white pixels in the mask
    area = np.count_nonzero(mask)
    # Determine presence based on a threshold
    presence = 1 if area >= threshold else 0
    return area, presence

def bcc_extract_features(img):
    result, mask = extract_ulceration_by_color(img)
    area, ul_presence = calculate_area_and_presence(mask)
    result, mask, total_area, ovoids_presence = extract_blue_gray_ovoids(img)
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_image, vessel_highlight, area, vessel_presence = extract_bw(original_image)
    return ul_presence, ovoids_presence, vessel_presence
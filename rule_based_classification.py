from configure import configure
from tensorflow.keras.models import load_model
import cv2
import numpy as np

def classify_bcc(arborizing_vessels, blue_gray_avoids, ulcerations):
    # Evaluate conditions based on the input features
    if arborizing_vessels == 0 and blue_gray_avoids == 0 and ulcerations == 0:
        return "unpredictable"
    elif arborizing_vessels == 0 and blue_gray_avoids == 0 and ulcerations == 1:
        return "Superficial BCC"
    elif arborizing_vessels == 0 and blue_gray_avoids == 1 and ulcerations == 0:
        return "pigmented"
    elif arborizing_vessels == 0 and blue_gray_avoids == 1 and ulcerations == 1:
        return "Pigmented or superficial"
    elif arborizing_vessels == 1 and blue_gray_avoids == 0 and ulcerations == 0:
        return "Nodular BCC"
    elif arborizing_vessels == 1 and blue_gray_avoids == 0 and ulcerations == 1:
        return "Nodular or Superficial"
    elif arborizing_vessels == 1 and blue_gray_avoids == 1 and ulcerations == 0:
        return "Nodular or Pigmented"
    elif arborizing_vessels == 1 and blue_gray_avoids == 1 and ulcerations == 1:
        return "unpredictable"
    else:
        return "Unknown case"

def classify_scc(visual_dots, rosettes, white_circles):
    # Evaluate conditions based on the input features
    if visual_dots == 1 and rosettes == 1 and white_circles == 1:
        return "unpredictable"
    elif visual_dots == 1 and rosettes == 1 and white_circles == 0:
        return "Conventional SCC or Keratoacanthoma"
    elif visual_dots == 1 and rosettes == 0 and white_circles == 1:
        return "Conventional SCC or Bowen's Disease"
    elif visual_dots == 1 and rosettes == 0 and white_circles == 0:
        return "Conventional SCC"
    elif visual_dots == 0 and rosettes == 1 and white_circles == 1:
        return "Bowen's Disease SCC"
    elif visual_dots == 0 and rosettes == 1 and white_circles == 0:
        return "Keratoacanthoma-type SCC"
    elif visual_dots == 0 and rosettes == 0 and white_circles == 1:
        return "Bowen's Disease"
    elif visual_dots == 0 and rosettes == 0 and white_circles == 0:
        return "unpredictable"
    else:
        return "Unknown condition - please verify inputs"

def classify_melanoma(blue_white_veil, regression, asymmetry):
    # Evaluate conditions based on the input features
    if blue_white_veil == 1 and regression == 1 and asymmetry == 1:
        return "unpredictable"
    elif blue_white_veil == 1 and regression == 1 and asymmetry == 0:
        return "LMM and NM"
    elif blue_white_veil == 1 and regression == 0 and asymmetry == 1:
        return "NM, SSM"
    elif blue_white_veil == 0 and regression == 1 and asymmetry == 0:
        return "Lentigo Maligna Melanoma (LMM)"
    elif blue_white_veil == 1 and regression == 1 and asymmetry == 0:
        return "Superficial Spreading Melanoma (SSM)"
    elif blue_white_veil == 0 and regression == 1 and asymmetry == 1:
        return "Nodular Melanoma (NM)"
    elif blue_white_veil == 0 and regression == 0 and asymmetry == 0:
        return "unpredictable"
    else:
        return "Case not covered in the table"
    
def get_disease_name(type_index, class_index):
    # Define the lists of disease names
    names1 = ['Lentigo Maligna Melanoma (LMM)', 'Nodular Melanoma (NM)', 'Superficial spreading melanoma (SSM)']
    names2 = ['Nodular BCC', 'Superficial BCC', 'Pigmented BCC']
    names3 = ['Keratoacanthoma-type SCC', 'Conventional SCC', 'Bowens Disease']
    
    # Create a dictionary to map type_index to the corresponding names list
    disease_mapping = {
        0: names1,
        1: names2,
        2: names3
    }
    
    # Check if the provided type_index is valid
    if type_index in disease_mapping:
        names_list = disease_mapping[type_index]
        # Check if the provided class_index is valid
        if 0 <= class_index < len(names_list):
            return names_list[class_index]
        else:
            return "Invalid class index"
    else:
        return "Invalid type index"
    
def classify_disease(asymmetry_presence, blue_white_veil_presence, regression_structure_presence,
            ul_presence, ovoids_presence, vessel_presence,
            dotted_vessels_presence, white_follicles_presence, rosette_presence, color, symmetric, image):
    done, error = configure(symmetric, color)
    result = None
    predicted = None
    if done:
        result = error
        predicted = error
    else:
        print(symmetric)
        if symmetric == 0:
            result = classify_melanoma(blue_white_veil_presence, regression_structure_presence, asymmetry_presence)
        elif symmetric == 1:
            result = classify_bcc(vessel_presence, ovoids_presence, ul_presence)
        elif symmetric == 2:
            result = classify_scc(dotted_vessels_presence, rosette_presence, white_follicles_presence)
        model = None
        img= cv2.resize(image, (224, 224))
        img_array = np.expand_dims(img, axis=0)
        img_array = img_array / 255.0  # Normalize the image
        if symmetric == 0:
            model = load_model('./model/cnn_model_melonoma.h5')
        elif symmetric == 1:
            model = load_model('./model/cnn_model_bcc.h5')
        elif symmetric == 2:
            model = load_model('./model/cnn_model_scc.h5')
        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        predicted = get_disease_name(symmetric, int(predicted_class))
    return result, predicted
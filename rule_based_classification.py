from configure import configure

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
    
def classify_disease(asymmetry_presence, blue_white_veil_presence, regression_structure_presence,
            ul_presence, ovoids_presence, vessel_presence,
            dotted_vessels_presence, white_follicles_presence, rosette_presence, color, symmetric):
    done, error = configure(symmetric, color)
    print(done, error)
    result = None
    if done:
        result = error
    else:
        print(symmetric)
        if symmetric == 0:
            result = classify_melanoma(blue_white_veil_presence, regression_structure_presence, asymmetry_presence)
        elif symmetric == 1:
            result = classify_bcc(vessel_presence, ovoids_presence, ul_presence)
        elif symmetric == 2:
            print(symmetric, color)
            print(dotted_vessels_presence, rosette_presence, white_follicles_presence)
            result = classify_scc(dotted_vessels_presence, rosette_presence, white_follicles_presence)
    return result
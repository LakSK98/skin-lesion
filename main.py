from flask import Flask, request, jsonify
from features_extraction_melonoma import melonoma_extract_features
from features_extraction_bcc import bcc_extract_features
from features_extraction_scc import scc_extract_features
import pickle
import numpy as np
from io import BytesIO
from PIL import Image
from rule_based_classification import classify_disease
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# with open('model.pkl', 'rb') as file:
#     loaded_clf = pickle.load(file)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No any Image fiiles in your request."}), 400
    skin_lesion = int(request.args.get('skin_lesion'))
    cv2 = request.files["image"]
    if cv2.filename == "":
        return jsonify({"error":"Proide a valid file."}), 400
    else:
        color = cv2.filename
    image_stream = cv2.read()
    image = Image.open(BytesIO(image_stream)).convert('RGB')
    image = np.array(image)
    try:
        asymmetry_presence, blue_white_veil_presence, regression_structure_presence = melonoma_extract_features(image)
        ul_presence, ovoids_presence, vessel_presence = bcc_extract_features(image)
        dotted_vessels_presence, white_follicles_presence, rosette_presence = scc_extract_features(image)
        # prediction = loaded_clf.predict([features])
        result = classify_disease(
            asymmetry_presence, blue_white_veil_presence, regression_structure_presence,
            ul_presence, ovoids_presence, vessel_presence,
            dotted_vessels_presence, white_follicles_presence, rosette_presence, color, skin_lesion
            )
        return jsonify({"Disease": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
from flask import Flask, request, render_template
import numpy as np
import pickle
import os


model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)


CROP_DICT = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
    
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

       
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

      
        prediction = model.predict(final_features)[0]

        if prediction in CROP_DICT:
            crop = CROP_DICT[prediction]
            result = f"Recommended Crop: {crop}"

          
            crop_filename = crop.lower()
            possible_extensions = [".jpeg", ".jpg", ".png"]

            image_name = "default.png"  # fallback
            for ext in possible_extensions:
                test_path = os.path.join("static", crop_filename + ext)
                if os.path.exists(test_path):
                    image_name = crop_filename + ext
                    break
        else:
            result = "Sorry, crop could not be determined."
            image_name = "default.png"

    except Exception as e:
        result = f"Invalid input: {e}"
        image_name = "default.png"

    return render_template('index.html', result=result, image_name=image_name)


if __name__ == "__main__":
    app.run(debug=True)

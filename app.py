from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Elevation = float(request.form['Elevation(meters)'])
        Aspect = int(request.form['Aspect(degrees)'])
        Slope = float(request.form['Slope(degrees)'])
        Horizontal_Distance_To_Hydrology = float(request.form['Horizontal Distance To Hydrology(meters)'])
        Vertical_Distance_To_Hydrology = int(request.form['Vertical Distance To Hydrology(meters)'])
        Horizontal_Distance_To_Roadways = float(request.form['Horizontal Distance To Roadways(meters)'])
        Hillshade_9am = float(request.form['Hillshade 9am'])
        Hillshade_Noon = float(request.form['Hillshade Noon'])
        Hillshade_3pm = int(request.form['Hillshade 3pm'])
        Horizontal_Distance_To_Fire_Points = float(request.form['Horizontal Distance To Fire Points(meters)'])
        Soil_Type = request.form['Soil Type']
        if (Soil_Type == 'Cathedral Family'):
            Soil_Type = 0
        elif (Soil_Type == 'Bullwark(outcrop complex)'):
            Soil_Type = 1
        elif (Soil_Type == 'Bullwark(land complex)'):
            Soil_Type = 2
        elif (Soil_Type == 'Legault family'):
            Soil_Type = 3
        elif (Soil_Type == 'Catamount family'):
            Soil_Type = 4
        elif (Soil_Type == 'Pachic Argiborolis'):
            Soil_Type = 5
        elif (Soil_Type == 'Cryaquolis'):
            Soil_Type = 6
        elif (Soil_Type == 'Gateview family'):
            Soil_Type = 7
        elif (Soil_Type == 'Rogert family'):
            Soil_Type = 8
        elif (Soil_Type == 'Typic Cryaquolis(Borohemists complex)'):
            Soil_Type = 9
        elif (Soil_Type == 'Vanet'):
            Soil_Type = 10
        elif (Soil_Type == 'Typic Cryaquepts(Typic Cryaquolls complex)'):
            Soil_Type = 11
        elif (Soil_Type == 'Typic Cryaquolls(Leighcan family, till substratum complex)'):
            Soil_Type = 12
        elif (Soil_Type == 'Leighcan family(extremely boulder)'):
            Soil_Type = 13
        elif (Soil_Type == 'Leighcan family(Typic Cryaquolls complex)'):
            Soil_Type = 14
        elif (Soil_Type == 'Leighcan family(extremely stony)'):
            Soil_Type = 15
        elif (Soil_Type == 'Leighcan family(warm, extremely stony)'):
            Soil_Type = 16
        elif (Soil_Type == 'Granile'):
            Soil_Type = 17
        elif (Soil_Type == 'Leighcan family(warm - Rock outcrop complex, extremely stony)'):
            Soil_Type = 18
        elif (Soil_Type == 'Leighcan family(Rock outcrop complex, extremely stony)'):
            Soil_Type = 19
        elif (Soil_Type == 'Como(Legault families complex, extremely stony)'):
            Soil_Type = 20
        elif (Soil_Type == 'Haploborolis'):
            Soil_Type = 21
        elif (Soil_Type == 'Como family(Rock land - Legault family complex, extremely stony)'):
            Soil_Type = 22
        elif (Soil_Type == 'Leighcan(Catamount families complex, extremely stony)'):
            Soil_Type = 23
        elif (Soil_Type == 'Catamount family(Rock outcrop)'):
            Soil_Type = 24
        elif (Soil_Type == 'Leighcan(Catamount families - Rock outcrop complex, extremely stony)'):
            Soil_Type = 25
        elif (Soil_Type == 'Cryorthents'):
            Soil_Type = 26
        elif (Soil_Type == 'Cryumbrepts'):
            Soil_Type = 27
        elif (Soil_Type == 'Bross family'):
            Soil_Type = 28
        elif (Soil_Type == 'Rock outcrop'):
            Soil_Type = 29
        elif (Soil_Type == 'Leighcan'):
            Soil_Type = 30
        elif (Soil_Type == 'Moran family(Cryorthents - Leighcan family complex, extremely stony)'):
            Soil_Type = 31
        elif (Soil_Type == 'Ratake family'):
            Soil_Type = 32
        elif (Soil_Type == 'Moran family(Cryorthents - Rock land complex, extremely stony)'):
            Soil_Type = 33
        elif (Soil_Type == 'Vanet family(Rock outcrop complex, rubble)'):
            Soil_Type = 34
        elif (Soil_Type == 'Vanet(Wetmore families - Rock outcrop complex, stony)'):
            Soil_Type = 35
        elif (Soil_Type == 'Gothic family'):
            Soil_Type = 36
        elif (Soil_Type == 'Supervisor'):
            Soil_Type = 37
        else:
            Soil_Type = 39

        Wilderness_Area = request.form['Wilderness Area']
        if (Wilderness_Area == 'Rawah'):
            Wilderness_Area = 0
        elif (Wilderness_Area == 'Neota'):
            Wilderness_Area = 1
        elif (Wilderness_Area == 'Comanche Peak'):
            Wilderness_Area = 2
        else:
            Wilderness_Area = 3

        prediction= model.predict([[Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology,
                                     Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am,
                                     Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area,
                                     Soil_Type]])
        le_name_mapping = {'Type 1': 0, 'Type 2': 1, 'Type 3': 2, 'Type 4': 3,
                           'Type 5': 4, 'Type 6': 5, 'Type 7': 6, }
        def get_key(val):
            for key, value in le_name_mapping.items():
                if val == value:
                    return key

        result = get_key(prediction[0])
        output = result

        return render_template('index.html', cover_type='Predicted forest cover type is {}'.format(output))


if __name__ == "__main__":
    app.run(port=5001,debug=True)

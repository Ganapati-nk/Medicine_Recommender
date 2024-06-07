from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
from nltk import PorterStemmer


# flask app
app = Flask(__name__)



# load databasedataset===================================
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")


# load model===========================================
svc = pickle.load(open('svc.pkl','rb'))


#============================================================
# custome and helping functions
#==========================helper funtions================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout

symptoms_dict = {'itch': 0, 'skin rash': 1, 'nodal skin erupt': 2, 'continu sneez': 3, 'shiver': 4, 'chill': 5, 'joint pain': 6, 'stomach pain': 7, 'acid': 8, 'ulcer on tongu': 9, 'muscl wast': 10, 'vomit': 11, 'burn micturit': 12, 'spot  urin': 13, 'fatigu': 14, 'weight gain': 15, 'anxieti': 16, 'cold hand and feet': 17, 'mood swing': 18, 'weight loss': 19, 'restless': 20, 'lethargi': 21, 'patch in throat': 22, 'irregular sugar level': 23, 'cough': 24, 'high fever': 25, 'sunken eye': 26, 'breathless': 27, 'sweat': 28, 'dehydr': 29, 'indigest': 30, 'headach': 31, 'yellowish skin': 32, 'dark urin': 33, 'nausea': 34, 'loss of appetit': 35, 'pain behind the eye': 36, 'back pain': 37, 'constip': 38, 'abdomin pain': 39, 'diarrhoea': 40, 'mild fever': 41, 'yellow urin': 42, 'yellow of eye': 43, 'acut liver failur': 44, 'fluid overload': 45, 'swell of stomach': 46, 'swell lymph node': 47, 'malais': 48, 'blur and distort vision': 49, 'phlegm': 50, 'throat irrit': 51, 'red of eye': 52, 'sinu pressur': 53, 'runni nose': 54, 'congest': 55, 'chest pain': 56, 'weak in limb': 57, 'fast heart rate': 58, 'pain dure bowel movement': 59, 'pain in anal region': 60, 'bloodi stool': 61, 'irrit in anu': 62, 'neck pain': 63, 'dizzi': 64, 'cramp': 65, 'bruis': 66, 'obes': 67, 'swollen leg': 68, 'swollen blood vessel': 69, 'puffi face and eye': 70, 'enlarg thyroid': 71, 'brittl nail': 72, 'swollen extremeti': 73, 'excess hunger': 74, 'extra marit contact': 75, 'dri and tingl lip': 76, 'slur speech': 77, 'knee pain': 78, 'hip joint pain': 79, 'muscl weak': 80, 'stiff neck': 81, 'swell joint': 82, 'movement stiff': 83, 'spin movement': 84, 'loss of balanc': 85, 'unsteadi': 86, 'weak of one bodi side': 87, 'loss of smell': 88, 'bladder discomfort': 89, 'foul smell of urin': 90, 'continu feel of urin': 91, 'passag of gase': 92, 'intern itch': 93, 'toxic look (typhos)': 94, 'depress': 95, 'irrit': 96, 'muscl pain': 97, 'alter sensorium': 98, 'red spot over bodi': 99, 'belli pain': 100, 'abnorm menstruat': 101, 'dischromic  patch': 102, 'water from eye': 103, 'increas appetit': 104, 'polyuria': 105, 'famili histori': 106, 'mucoid sputum': 107, 'rusti sputum': 108, 'lack of concentr': 109, 'visual disturb': 110, 'receiv blood transfus': 111, 'receiv unsteril inject': 112, 'coma': 113, 'stomach bleed': 114, 'distent of abdomen': 115, 'histori of alcohol consumpt': 116, 'fluid overload.1': 117, 'blood in sputum': 118, 'promin vein on calf': 119, 'palpit': 120, 'pain walk': 121, 'pu fill pimpl': 122, 'blackhead': 123, 'scur': 124, 'skin peel': 125, 'silver like dust': 126, 'small dent in nail': 127, 'inflammatori nail': 128, 'blister': 129, 'red sore around nose': 130, 'yellow crust ooz': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]




# creating routes========================================


@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        
        # Check if the user entered any symptoms
        if not symptoms or symptoms == "Symptoms":
            message = "Please enter symptoms."
            return render_template('index.html', message=message)
        
        # Split the user's input into a list of symptoms (assuming they are comma-separated)
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        
        # Stem the symptoms
        stemmer = PorterStemmer()
        stemmed_symptoms = [stemmer.stem(symptom) for symptom in user_symptoms]
        
        # Filter out symptoms that are not in the symptoms_dict
        valid_symptoms = [symptom for symptom in stemmed_symptoms if symptom in symptoms_dict]
        
        # Check if any valid symptoms are left
        if not valid_symptoms:
            message = "We couldn't find any matching symptoms in our database. Please double-check your input and try again, or consult a healthcare professional for personalized advice."

            return render_template('index.html', message=message)
        
        # Get predicted disease
        predicted_disease = get_predicted_value(valid_symptoms)
        
        # Get additional information based on the predicted disease
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
        
        # Prepare precautions for display
        my_precautions = [precaution for precaution_list in precautions for precaution in precaution_list]
        
        # Render the template with the predicted disease and additional information
        return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                               my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                               workout=workout)
    
    return render_template('index.html')



# about view funtion and path
@app.route('/about')
def about():
    return render_template("about.html")
# contact view funtion and path
@app.route('/contact')
def contact():
    return render_template("contact.html")

# developer view funtion and path
@app.route('/developer')
def developer():
    return render_template("developer.html")

# about view funtion and path
@app.route('/blog')
def blog():
    return render_template("blog.html")


if __name__ == '__main__':

    app.run(debug=True)
import os
import joblib
import pandas as pd

# Ensure the script runs from the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_path = os.path.join(script_dir, "artifacts")

# Try loading models and scalers with error handling
try:
    model_young = joblib.load(os.path.join(artifacts_path, "model_young.joblib"))
    model_rest = joblib.load(os.path.join(artifacts_path, "model_rest.joblib"))
    scaler_young = joblib.load(os.path.join(artifacts_path, "scaler_young.joblib"))
    scaler_rest = joblib.load(os.path.join(artifacts_path, "scaler_rest.joblib"))
except FileNotFoundError as e:
    print(f"Error: {e}\nCheck if model and scaler files exist in the 'artifacts' directory.")
    exit(1)  # Exit script if files are missing


def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)

    max_score = 14  # Max risk score (heart disease + another high-risk condition)
    min_score = 0  # No disease

    return (total_risk_score - min_score) / (max_score - min_score)


def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    mapping = {
        'Gender': {'Male': 'gender_Male'},
        'Region': {'Northwest': 'region_Northwest', 'Southeast': 'region_Southeast', 'Southwest': 'region_Southwest'},
        'Marital Status': {'Unmarried': 'marital_status_Unmarried'},
        'BMI Category': {'Obesity': 'bmi_category_Obesity', 'Overweight': 'bmi_category_Overweight',
                         'Underweight': 'bmi_category_Underweight'},
        'Smoking Status': {'Occasional': 'smoking_status_Occasional', 'Regular': 'smoking_status_Regular'},
        'Employment Status': {'Salaried': 'employment_status_Salaried',
                              'Self-Employed': 'employment_status_Self-Employed'},
    }

    for key, value in input_dict.items():
        if key in mapping and value in mapping[key]:
            df[mapping[key][value]] = 1
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key in ['Age', 'Number of Dependants', 'Income in Lakhs', 'Genetical Risk']:
            df[key.lower().replace(" ", "_")] = value

    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    df = handle_scaling(input_dict['Age'], df)

    return df


def handle_scaling(age, df):
    scaler_object = scaler_young if age <= 25 else scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = 0  # Placeholder column to match scaler expectations
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop('income_level', axis=1, inplace=True)

    return df


def predict(input_dict):
    input_df = preprocess_input(input_dict)

    model = model_young if input_dict['Age'] <= 25 else model_rest
    prediction = model.predict(input_df)

    return int(prediction[0])


# Test the function (Example Input)
if __name__ == "__main__":
    sample_input = {
        "Age": 28,
        "Number of Dependants": 2,
        "Income in Lakhs": 5,
        "Insurance Plan": "Gold",
        "Genetical Risk": 3,
        "Medical History": "diabetes & high blood pressure",
        "Gender": "Male",
        "Region": "Northwest",
        "Marital Status": "Unmarried",
        "BMI Category": "Obesity",
        "Smoking Status": "Regular",
        "Employment Status": "Salaried"
    }

    print("Predicted Insurance Category:", predict(sample_input))

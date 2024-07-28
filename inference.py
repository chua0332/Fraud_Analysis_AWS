import os
import joblib
import pandas as pd
import argparse
from datetime import datetime
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

# Function to calculate age
#born has to be a datetime object
def calculate_age(born):
    today = datetime.today()
    return int(today.year - born.year)

def preprocess_data(data):
    # Handle missing values
     #Converting dob to a proper datetime yyyymmdd date object
    data['dob'] = pd.to_datetime(data['dob'], format='%Y-%m-%d')
    #Calculating for the age variable
    data['age'] = data['dob'].apply(calculate_age)

    # Now lets select columns that I think are possibly useful
    selectedcolumns = ['category','amt','gender','job','merchant', 'city', 'state', 'age']
    data = data[selectedcolumns]

    return data

def load_model(model_path):
    return joblib.load(model_path)

def main(args):
    environment = os.getenv("ENVIRONMENT", "local")
    test_data_path = os.path.join(args.inference, "fraud_test.csv")
    model_path = "/opt/ml/model/model.joblib"
    preprocessor_path = "/opt/ml/model/preprocessor.joblib"
    output_path = "/opt/ml/output/predictions.csv"

    if environment == "local":
        test_data_path = "data/inference/fraud_test.csv"
        model_path = "model/model.joblib"
        preprocessor_path = "model/preprocessor.joblib"
        output_path = "output/predictions.csv"

    # Load model
    # model = load_model(model_path)
    model = joblib.load(model_path)
    #print(model_columns)
    test_data = pd.read_csv(test_data_path)

     #Basic prep to only select columns we need
    test_data_processed = preprocess_data(test_data)

    #Doing preprocessing to shape the data into right format required for ML classification
    preprocessor = joblib.load(preprocessor_path)
    test_data_processed = preprocessor.transform(test_data_processed)

    predictions = model.predict(test_data_processed)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    pd.DataFrame(predictions, columns=["is_fraud"]).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference",
        type=str,
        help="Path to the inference data",
        default="/opt/ml/input/data/inference",
    )
    parser.add_argument("--n-estimators", type=int, default=100)

    args = parser.parse_args()
    main(args)

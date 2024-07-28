import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from datetime import datetime
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def main(args):
    # Load training data
    train_data_path = os.path.join(args.train, "fraud_train.csv")
    output_dir = "/opt/ml/model"

    environment = os.getenv("ENVIRONMENT", "local")

    if environment == "local":
        train_data_path = "data/train/fraud_train.csv"
        output_dir = "output"

    data = pd.read_csv(train_data_path)

    #Converting dob to a proper datetime yyyymmdd date object
    data['dob'] = pd.to_datetime(data['dob'], format='%Y-%m-%d')
    #Calculating for the age variable
    data['age'] = data['dob'].apply(calculate_age)

    # Now lets select columns that I think are possibly useful
    selectedcolumns = ['category','amt','gender','job','merchant', 'city', 'state', 'age', 'is_fraud']
    data = data[selectedcolumns]

    #let us start separating the data into x & y
    X = data.drop(columns=['is_fraud'],axis=1)
    y = data['is_fraud']

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    #Selecting the numerical and categorical columns for easy preprocessing!
    num_columns = X.select_dtypes(include=['float64','int64']).columns
    cat_columns = X.select_dtypes(include=['object']).columns

    #Creating the preprocessing pipelines!
    cat_transformer = make_pipeline(OneHotEncoder(handle_unknown='ignore',sparse_output=False))
    num_transformer = make_pipeline(MinMaxScaler())

    preprocessor = make_pipeline(ColumnTransformer([('num_transformer',num_transformer,num_columns),('cat_transformer',cat_transformer,cat_columns)],
                                               remainder='passthrough'))
    
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)

    # Train model
    model = RandomForestClassifier(n_estimators=args.n_estimators)
    model.fit(X_train, y_train)

    # Validate model
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    print(f"Validation Accuracy: {accuracy}")

    recall = recall_score(y_val, predictions)
    print(f"Validation Recall: {recall}")

    precision = precision_score(y_val, predictions)
    print(f"Validation Precision {precision}")

    # Save model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save model
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Function to calculate age
#born has to be a datetime object
def calculate_age(born):
    today = datetime.today()
    return int(today.year - born.year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        help="Path to the training data",
        default="/opt/ml/input/data/train",
    )
    parser.add_argument("--n-estimators", type=int, default=100)

    args = parser.parse_args()
    main(args)

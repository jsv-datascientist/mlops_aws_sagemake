
from sagemaker.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
import pandas as pd
import numpy as np
import os
import argparse



def load_model(model_dir):
    model = load(os.path.join(model_dir , "model.joblib"))
    return model 


def _parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum depth of the tree")  
    parser.add_argument("--model-dir", type="str")
    parser.add_argument("--train", type="str", default = os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument("--test", type="str", default = os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument("--train-file", type="str", default ="train.csv")
    parser.add_argument("--test-file", type="str", default ="test.csv")
    return parser.parse_args()
    


if __name__ == "__main__":
    
    args = _parse_args()
    print(args)
    train_path = os.path.join(args.train, args.train_file)
    test_path = os.path.join(args.test, args.test_file)

    # Load the data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Split the data into features and labels
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]
    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    # Train the model
    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
    model.fit(X_train, y_train)

    # Save the model
    dump(model, os.path.join(args.model_dir, "model.joblib"))

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    
    
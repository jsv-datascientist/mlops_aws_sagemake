
import pandas as pd
import numpy as np 
import argparse
import os

from joblib import dump, load 

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split    


def _parse_args():
    
    parser = argparse.ArgumentParser(description="Preprocess the data")
    parser.add_argument(
        "--filepath", type=str, default="data/raw/", help="Path to the data"
    )
    parser.add_argument(
        "--filename", type=str, default="data.csv", help="Name of the data file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed/", help="Path to the output directory"
    )
    parser.add_argument(
        "--top_k", type=int, default=20, help="Top k features to select"
    )
    
    return parser.parse_args()


def get_top_k_features(X, y, top_k):
    """
    Get the top k features using ExtraTreesClassifier
    """
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    feature_df = pd.DataFrame(data=(X.columns, model.feature_importances_)).T.sort_values(by=1, ascending=False)
    cols = feature_df.head(k)[0].values
    
    return cols


if __name__=="__main__":
    args,_ = _parse_args()
    
    df = pd.read_csv(os.path.join(args.filepath, args.filename))
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Get top k features
    top_k_features = get_top_k_features(X_train, y_train, args.top_k)
    # Select top k features
    X_train = X_train[top_k_features]
    X_test = X_test[top_k_features]
    # Save the data     
    os.makedirs(args.output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(args.output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(args.output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(args.output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(args.output_dir, "y_test.csv"), index=False)
    
    
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    
    '''# Encode categorical features
    for col in categorical_features:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
    '''
    le = LabelEncoder()
    X_train[categorical_features] = X_train[categorical_features].apply(le.fit_transform)
    X_test[categorical_features] = X_test[categorical_features].apply(le.transform) 
    # Save the label encoder
    dump(le, os.path.join(args.output_dir, "label_encoder.joblib"))
    # Save the data
    os.makedirs(args.output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(args.output_dir, "X_train_encoded.csv"), index=False)
    X_test.to_csv(os.path.join(args.output_dir, "X_test_encoded.csv"), index=False)
    y_train.to_csv(os.path.join(args.output_dir, "y_train_encoded.csv"), index=False)
    y_test.to_csv(os.path.join(args.output_dir, "y_test_encoded.csv"), index=False)
    # Save the top k features
    with open(os.path.join(args.output_dir, "top_k_features.txt"), "w") as f:
        for feature in top_k_features:
            f.write(f"{feature}\n")
    # Save the target
    y = pd.DataFrame(y)
    y.to_csv(os.path.join(args.output_dir, "target.csv"), index=False)
    
    print("Preprocessing complete")
    print(f"Top {args.top_k} features: {top_k_features}")
    print(f"Data saved to {args.output_dir}")
    print(f"Label encoder saved to {os.path.join(args.output_dir, 'label_encoder.joblib')}")
    print(f"Top k features saved to {os.path.join(args.output_dir, 'top_k_features.txt')}")
    print(f"Target saved to {os.path.join(args.output_dir, 'target.csv')}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

   
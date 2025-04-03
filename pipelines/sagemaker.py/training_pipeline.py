from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn

sklearn_estimator = SKLearn(
    entry_point="/training/training.py"
    role = get_execution_role()
    instance_count=1,
    instance_type="ml.m5.large",
    volume_size=30,
    framework_version="0.23-1",
    base_job_name="sklearn-training",
    metrics_definitions=[
        {"Name": "accuracy", "Regex": "accuracy: ([0-9\\.]+)"},
        {"Name": "f1", "Regex": "f1: ([0-9\\.]+)"}
    ]
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "test-file": "validation.csv",
        "train-file": "train.csv",
        "model-dir": "/opt/ml/model",
        "output-dir": "/opt/ml/output",
        "log-dir": "/opt/ml/output/logs",
        "model-name": "sklearn-model",
        "model-version": "1.0",
    },
)

sklearn_estimator.fit({"train": s3_input_train,
                       "validation": s3_input_validation}
)
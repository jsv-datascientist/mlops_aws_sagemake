from sagemaker.tuner import IntegerParameter

hypeparameter_ranges = {
    "n-estimators": IntegerParameter(50, 200),
    "max-depth": IntegerParameter(3, 10),
    "min-samples-leaf": IntegerParameter(2, 6)
}


Optimizer = sagemaker.tuner.HyperparameterTuner(
    estimator=sklearn_estimator,
    hypeparameter_ranges=hypeparameter_ranges,
    objective_metric_name="accuracy",
    objective_type="Maximize",
    max_jobs=10,
    max_parallel_jobs=2,
    metric_definitions=[
        {"Name": "accuracy", "Regex": "accuracy: ([0-9\\.]+)"}
    ],
    base_tuning_job_name="hyperparameter-tuning-job"
)

Optimizer.fit({"train": s3_input_train, "test": s3_input_test})


results = Optimizer.analytics().dataframe()
while results.empty:
    time.sleep(1)
    results = Optimizer.analytics().dataframe()
results.head()


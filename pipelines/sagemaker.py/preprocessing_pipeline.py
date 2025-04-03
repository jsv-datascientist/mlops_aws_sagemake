
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker import get_execution_role


sklearn_processor = SKLearnProcessor(
    role=get_execution_role(),
    instance_type="ml.m5.large",    # Change to your instance type
    instance_count=1,   # Change to your instance count
    volume_size_in_gb=30,  # Change to your volume size
    max_runtime_in_seconds=1200,  # Change to your max runtime
    framework_version="0.23-1",  # Change to your framework version
    base_job_name="sklearn-processing-job",  # Change to your base job name
    sagemaker_session=None,  # Change to your sagemaker session if needed
)

sklearn_processor.run(
    code = "/preprocessing/preprocessing.py",  # Change to your preprocessing script path
    inputs=[
        ProcessingInput(
            source="s3://your-bucket/input-data",  # Change to your input data S3 path
            destination="/processing/input",
            s3_input_mode="File",
            s3_data_distribution_type="SharedByS3Key"
        )
    ],
    outputs=[
        ProcessingOutput( output_name="train_data", source="/processing/output/train/", destination="s3://your-bucket/output-data" ), # Change to your output data S3 path
        ProcessingOutput( output_name="test_data", source="/processing/output/test", destination="s3://your-bucket/output-data" ),
        ProcessingOutput( output_name="validation_data", source="/processing/output/validation", destination="s3://your-bucket/output-data" )
        
    ],
)
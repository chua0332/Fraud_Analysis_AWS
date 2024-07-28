import boto3
import datetime

# Define AWS credentials and region
region = "ap-southeast-1"
role_arn = "arn:aws:iam::791934322985:role/sagemaker-eugene"

# Define ECR image and S3 paths
ecr_image = "791934322985.dkr.ecr.ap-southeast-1.amazonaws.com/fraud_mlops:latest"
s3_input_train = "s3://fraud-mlops/data/train"
s3_output_path = "s3://fraud-mlops/output"

# Initialize Boto3 SageMaker client
sagemaker_client = boto3.client("sagemaker", region_name=region)

# Define training job name
training_job_name = (
    f'custom-training-job-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
)

# Create training job configuration
training_job_config = {
    "TrainingJobName": training_job_name,
    "AlgorithmSpecification": {"TrainingImage": ecr_image, "TrainingInputMode": "File"},
    "RoleArn": role_arn,
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_input_train,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "ContentType": "text/csv",
            "InputMode": "File",
        }
    ],
    "OutputDataConfig": {"S3OutputPath": s3_output_path},
    "ResourceConfig": {
        "InstanceType": "ml.m5.large",
        "InstanceCount": 1,
        "VolumeSizeInGB": 50,
    },
    "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
    "HyperParameters": {"n_estimators": "100"},
}

# Create the training job
response = sagemaker_client.create_training_job(**training_job_config)

# Print the response
print(response)

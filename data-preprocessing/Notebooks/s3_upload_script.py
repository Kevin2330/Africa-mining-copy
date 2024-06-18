# This script allows me to upload the data to s3 

import boto3
from botocore import UNSIGNED
from botocore.client import Config

# Initialize the S3 client
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Define your bucket name
bucket_name = 'africamining'

# Define the file path and object key
file_path = '/Users/kd6801/Desktop/complete_data_y0.csv'  
object_key = 'complete_data_y0.csv'

# Open the file and upload it
with open(file_path, 'rb') as data:
    s3.put_object(Bucket=bucket_name, Key=object_key, Body=data)

print(f'File {object_key} has been uploaded to bucket {bucket_name}')


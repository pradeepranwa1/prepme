import boto3
from botocore.exceptions import NoCredentialsError
from botocore.exceptions import ClientError

AWS_S3_BUCKET = "prepme-ai"  
AWS_REGION = "ap-south-1"  

# Initialize S3 client
s3_client = boto3.client("s3")

def upload_file_to_s3(file_obj, object_name: str):
    """Uploads a file to S3 and returns the file URL"""
    try:
        s3_client.upload_fileobj(file_obj, AWS_S3_BUCKET, object_name)
        return f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/books/{object_name}"
    except NoCredentialsError:
        return "AWS credentials not found!"
    
def check_file_in_s3(file_key: str) -> bool:
    """
    Check if a file exists in the specified S3 bucket.
    :param file_key: The key (path) of the file in the S3 bucket.
    :return: True if the file exists, False otherwise.
    """
    try:
        s3_client.head_object(Bucket=AWS_S3_BUCKET, Key=file_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise  # Re-raise the exception if it's not a 404 error

def generate_presigned_url(file_name: str, expiration: int = 3600):
    """Generates a pre-signed URL for downloading a file"""
    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": AWS_S3_BUCKET, "Key": f"books/{file_name}"},
            ExpiresIn=expiration,
        )
        return url
    except Exception as e:
        return str(e)

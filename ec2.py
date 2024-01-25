import boto3

aws_access_key_id = ''
aws_secret_access_key = ''
aws_region = 'ap-south-1'

instance_type = 't2.micro'
image_id = 'ami-187760-linux'
key_name = 'bharu-test'

ec2 = boto3.client('ec2', region_name=aws_region, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

instance_params = {
    'ImageId': image_id,
    'InstanceType': instance_type,
    'KeyName': key_name,
    'MinCount': 1,
    'MaxCount': 1
}

response = ec2.run_instances(**instance_params)

instance_id = response['Instances'][0]['InstanceId']

print(f"Instance {instance_id} is being created.")

from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorchModel
import sagemaker

model_name = 'newdqn-trade-model'
# image_uri = '785573368785.dkr.ecr.us-east-1.amazonaws.com/sagemaker-inference-pytorch:2.0-cpu-py310'
s3_model_uri = 's3://sagemaker-us-east-1-842676017389/model2.tar.gz'

# Create a local SageMaker session
sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

# Create SageMaker model
role = 'arn:aws:iam::842676017389:role/blockhouse'
model = PyTorchModel(
    model_data=s3_model_uri,
    role=role,
    entry_point='inference.py',
    source_dir='code',
    framework_version='2.0',
    py_version='py310',
    sagemaker_session=sagemaker_session,
    name=model_name,
    container_log_level=10
)

# Deploy model locally
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='local',
    endpoint_name='newdqn-5-endpoint-local'
)

# Test the local endpoint

from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

predictor = Predictor(
    endpoint_name='newdqn-5-endpoint-local',
    sagemaker_session=sagemaker_session,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

# Replace this with your actual input data structure
test_data = {
    "ticker": "AAPL",
    "shares": 1000,
    "time_horizon": 390
}
result = predictor.predict(test_data)
print("Prediction result:", result)




# Replace this with your actual input data
# test_data = {"input": "your test input here"}
# result = predictor.predict(test_data)
# print("Prediction result:", result)

# # Remember to delete the endpoint after you're done
# predictor.delete_endpoint()
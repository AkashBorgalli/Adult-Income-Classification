import azureml
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core import Model
import json
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()
service = azureml.core.Webservice(workspace=ws, name='adult-income-service')
x_new = [[32,3,8536,5,4,2,7,1,2,0,0,0,15,22]]
print ('Adult Details: {}'.format(x_new[0]))
# Convert the array to a serializable list in a JSON document
input_json = json.dumps({"data": x_new})
# Call the web service, passing the input data (the web service will also accept the data in binary format)
predictions = service.run(input_data = input_json)
# Get the predicted class - it'll be the first (and only) one.
predicted_classes = json.loads(predictions)
print('prediction for the given adult details: ',predicted_classes[0])
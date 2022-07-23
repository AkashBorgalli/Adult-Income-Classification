from azureml.core import Datastore
from azureml.pipeline.core import Schedule
from azureml.core import Workspace
from azureml.pipeline.core import PublishedPipeline


ws = Workspace.from_config()
published_pipeline = PublishedPipeline.get(workspace=ws, id="03e3308b-2869-4ecd-81b3-e3b7c725e79a")
print(published_pipeline.name, published_pipeline.version)
training_datastore = Datastore(workspace=ws, name='workspaceblobstore')
pipeline_schedule = Schedule.create(ws, name='Reactive Training',
                                    description='trains model on data change',
                                    pipeline_id=published_pipeline.id,
                                    experiment_name='Training_Pipeline',
                                    datastore=training_datastore,
                                    path_on_datastore='dataset/salary_classification')

print("Pipeline schedule created.")
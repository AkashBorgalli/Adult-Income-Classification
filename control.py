
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep
from azureml.core import Environment
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Dataset, Workspace
from azureml.data.datapath import DataPath
import azureml.core
from azureml.core import Experiment, ScriptRunConfig, Environment,Workspace
from azureml.core.runconfig import DockerConfiguration
from azureml.widgets import RunDetails
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline
from azureml.widgets import RunDetails
import os, shutil
# Create a folder for the experiment files
experiment_folder = './'

# Load the workspace from the saved config file
ws = Workspace.from_config()

# creating env
from azureml.core import Experiment, ScriptRunConfig, Environment
experiment_env = Environment.from_conda_specification("adult_classification_env", "D:/Adult Classification Project/Adult-Income-Classification/environment.yml")

# Register the environment 
experiment_env.register(workspace=ws)
registered_env = Environment.get(ws, 'adult_classification_env')

# get compute cluster
pipeline_cluster = ComputeTarget(workspace=ws, name='classification-cpu-cluster')

# Create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()

# Use the compute you created above. 
pipeline_run_config.target = pipeline_cluster

# Assign the environment to the run configuration
pipeline_run_config.environment = registered_env

# Get the training dataset
adult_income_ds = ws.datasets.get("salary_classification")

# Create an OutputFileDatasetConfig (temporary Data Reference) for data passed from step 1 to step 2
prepped_data = OutputFileDatasetConfig("prepped_data")

# Step 1, Run the data prep script
analyse_step = PythonScriptStep(name = "Data Stats",
                                source_directory = experiment_folder,
                                script_name = "data_stats.py",
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)
prep_step = PythonScriptStep(name = "Prepare Data",
                                source_directory = experiment_folder,
                                script_name = "prep_adult_income.py",
                                arguments = ['--input-data', adult_income_ds.as_named_input('raw_data'),
                                             '--prepped-data', prepped_data],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)

# Step 3, run the training script
train_step = PythonScriptStep(name = "Normalize,Split,Train and Register Model",
                                source_directory = experiment_folder,
                                script_name = "train_adult_salary.py",
                                arguments = ['--training-data', prepped_data.as_input()],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)

print("Pipeline steps defined")

# Construct the pipeline
pipeline_steps = [analyse_step,prep_step, train_step]
pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
print("Pipeline is built.")

# Create an experiment and run the pipeline
experiment = Experiment(workspace=ws, name = 'adult-income-pipeline')
pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)
print("Pipeline submitted for execution.")
##RunDetails(pipeline_run).show()
pipeline_run.wait_for_completion(show_output=False)


from datetime import datetime

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from azure.ai.ml import MLClient
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import (
    JobSchedule,
    CronTrigger
)
from azureml.pipeline.core import TimeZone

# load component functions
from src.components.prep.prep_component import prepare_data_component
from src.components.train.train_component import train_component
from src.components.register.register_component import register_component


# -- Get a handle to Azure workspace --
try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

ml_client = MLClient.from_config(credential=credential)


# -- Set credentials
# -- We need to set and pass following credentials for registering model. Registering model requires
# workspace handle ml_client. Can't pass it as argument, because component function doesn't accept complex type.
# Getting handle using config is not possible, because it is running inside the job and config file doesn't exist there --
# TODO: read these from secrets or find some other solution not to keep them openly.
subscription_id = "<SUBSCRIPTION_ID>"
resource_group_name = "<RESOURCE_GROUP>"
workspace_name = "<WORKSPACE>"

# -- Retrieve an already attached Azure ML compute instance --
cpu_compute_target = "<compute_name>"
# print(ml_client.compute.get(cpu_compute_target))


# -- Define pipeline --
@pipeline(
    default_compute=cpu_compute_target,
)
def stock_predictor_pipeline_func(previous_time_window, next_time_window, test_size, registered_model_name):
    prepare_data_node = prepare_data_component(previous_time_window=previous_time_window, next_time_window=next_time_window, test_size=test_size)

    train_node = train_component(input_data=prepare_data_node.outputs.training_data)

    register_component(input_data=train_node.outputs.output_model,
                       registered_model_name=registered_model_name,
                       subscription_id=subscription_id,
                       resource_group_name=resource_group_name,
                       workspace_name=workspace_name)

pipeline_job = stock_predictor_pipeline_func(previous_time_window=10,
                                             next_time_window=5,
                                             test_size=0.2,
                                             registered_model_name="pipeline_stock_predictor",)


# -- Uncomment following 2 lines and run to execute pipeline job manually --
# pipeline_job = ml_client.jobs.create_or_update(pipeline_job, experiment_name="stock_predictor_full_pipeline")
# pipeline_job

# -- Set up scheduler to run pipeline job with custom input params at every run
pipeline_job = stock_predictor_pipeline_func(
    previous_time_window=10,
    next_time_window=5,
    test_size=0.2,
    registered_model_name="pipeline_stock_predictor",
)

pipeline_job.settings.default_compute = cpu_compute_target

schedule_name = "stock_predictor_scheduler"

schedule_start_time = datetime.utcnow()
cron_trigger = CronTrigger(
    expression="0 0 * * *",
    start_time=schedule_start_time,  # start time
    time_zone=TimeZone.CentralEuropeanStandardTime,  # time zone of expression
)

job_schedule = JobSchedule(
    name=schedule_name, trigger=cron_trigger, create_job=pipeline_job
)

job_schedule = ml_client.schedules.begin_create_or_update(schedule=job_schedule).result()
print(job_schedule)
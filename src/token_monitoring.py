from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    MonitorSchedule,
    CronTrigger,
    MonitorDefinition,
    ServerlessSparkCompute,
    MonitoringTarget,
    AlertNotification,
    GenerationTokenStatisticsSignal,
)
from azure.ai.ml.entities._inputs_outputs import Input
from azure.ai.ml.constants import MonitorTargetTasks, MonitorDatasetContext

# Authentication package
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()

# Update your Azure resources details
subscription_id = "INSERT YOUR SUBSCRIPTION ID"
resource_group = "INSERT YOUR RESOURCE GROUP NAME"
project_name = "INSERT YOUR PROJECT NAME" # This is the same as your Foundry project name
endpoint_name = "INSERT YOUR ENDPOINT NAME" # This is your deployment name without the suffix (e.g., deployment is "contoso-chatbot-1", endpoint is "contoso-chatbot")
deployment_name = "INSERT YOUR DEPLOYMENT NAME"

# These variables can be renamed but it is not necessary
monitor_name ="gen_ai_monitor_tokens" 
defaulttokenstatisticssignalname ="token-usage-signal" 

# Determine the frequency to run the monitor, and the emails to receive email alerts
trigger_schedule = CronTrigger(expression="15 10 * * *")
notification_emails_list = ["test@example.com", "def@example.com"]

ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=project_name,
)

spark_compute = ServerlessSparkCompute(instance_type="standard_e4s_v3", runtime_version="3.3")
monitoring_target = MonitoringTarget(
    ml_task=MonitorTargetTasks.QUESTION_ANSWERING,
    endpoint_deployment_id=f"azureml:{endpoint_name}:{deployment_name}",
)

# Create an instance of a token statistic signal
token_statistic_signal = GenerationTokenStatisticsSignal()

monitoring_signals = {
    defaulttokenstatisticssignalname: token_statistic_signal,
}

monitor_settings = MonitorDefinition(
compute=spark_compute,
monitoring_target=monitoring_target,
monitoring_signals = monitoring_signals,
alert_notification=AlertNotification(emails=notification_emails_list),
)

model_monitor = MonitorSchedule(
    name = monitor_name,
    trigger=trigger_schedule,
    create_monitor=monitor_settings
)

ml_client.schedules.begin_create_or_update(model_monitor)
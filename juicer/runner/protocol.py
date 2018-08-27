# coding=utf-8

## Comm protocol between clients, JuicerServer and Minion ##

# EXECUTE messages request the execution of a new job
# {
#   'workflow_id': <workflow identifier>
#   'app_id': <app identifier, representing an workflow instance>
#   'job_id': <job identifier, representing a command submited to the app>
#   'type': 'execute'
#   'app_configs': <key value pairs as app resource and env configurations>
#   'workflow': <workflow that will be transpiled and executed>
# }
EXECUTE = 'execute'

# DELIVER messages request the delivery of a result (task_id)
# {
#   'workflow_id': <workflow identifier>
#   'app_id': <app identifier, representing an workflow instance>
#   'job_id': <job identifier, representing a command submited to the app>
#   'type': 'deliver'
#   'task_id': <identifier of the task result to deliver>
#   'output': <queue identifier for publishing results>
#   'port': <port for fetching results>
#   'app_configs': <key value pairs as app resource and env configurations>
#   'workflow': <workflow that will be transpiled and executed>
# }
DELIVER = 'deliver'

# TERMINATE messages request the termination of a minion
# NOTE: if 'job_id' is present then we will stop all current jobs and keep the
# app alive, i.e., its resource context
# {
#   'workflow_id': <workflow identifier>
#   'app_id': <app identifier, representing an workflow instance>
#   ['job_id': <job identifier, representing a command submited to the app>]
#   'type': 'terminate'
# }
TERMINATE = 'terminate'

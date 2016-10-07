
from workflow import Workflow
from IO import IO
import json
from juicer.spark.control import Spark


if __name__ == '__main__':

    # Read the parameters
    io = IO()

    # Create the workflow, sort the tasks and plot the graph (optional)
    workflow = Workflow(io.args['json'])
    workflow.read_json()
    workflow.sort_tasks()
    workflow.plot_workflow(io.args['graph_outfile'])

    if workflow.workflow['framework'].lower() == "spark":
        spark = Spark(io.args['outfile'], workflow.workflow['name'],
        workflow.sorted_tasks)
        spark.execution()

    #juicer_excution(workflow, task_sorted, args['outfile'])

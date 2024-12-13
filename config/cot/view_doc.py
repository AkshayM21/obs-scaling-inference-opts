"""
Understand what the `doc` objects look like for each Task.
"""

from lm_eval.tasks import TaskManager, get_task_dict

if __name__ == "__main__":
    tm = TaskManager()
    task = get_task_dict('winogrande', tm)

    print(tm)

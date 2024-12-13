from lm_eval.tasks import TaskManager, get_task_dict

def _test(task, doc=None):
    if doc is None:
        # print(task.eval_docs)
        doc = task.eval_docs[0]

    print(f'\n{task.task_name} -----')
    print(doc)
    print(task.doc_to_text(doc))
    print(task.doc_to_target(doc))

def test_winograd(lang='en'):
    cot_task = list(get_task_dict('xwinograd_cot', tm).values())[0][f'xwinograd_cot_{lang}']
    gen_task = list(get_task_dict('xwinograd_generate', tm).values())[0][f'xwinograd_generate_{lang}']

    _test(cot_task)
    _test(gen_task)

def test_standard(cot_name, gen_name):
    cot_task = get_task_dict(cot_name, tm)[cot_name]
    gen_task = get_task_dict(gen_name, tm)[gen_name]

    _test(cot_task)
    _test(gen_task)

def test_truthfulqa():
    test_standard('truthfulqa_cot', 'truthfulqa_generative')

def test_hellaswag():
    test_standard('hellaswag_cot', 'hellaswag_generate')

def test_arc():
    test_standard('arc_challenge_cot', 'arc_challenge_generate')

def test_winogrande():
    test_standard('winogrande_cot', 'winogrande_generate')

if __name__ == "__main__":
    tm = TaskManager(include_path='/home/ubuntu/obs-scaling-inference-opts/config/cot')

    # test_winograd(lang='fr')
    # test_truthfulqa()
    # test_hellaswag()
    test_arc()
    # test_winogrande()

from mmte.tasks.confaide import ConfAIde_Task


def test_method():
    cfg = {
        'method_cfg': {
            'img_dir': './tmp_dir2',
            'img_size': (50, 50),
            'lazy_mode': True,
        }
    }
    task = ConfAIde_Task(
        task_id='confaide-task',
        dataset_id=dataset_ids[0],
        model_id='llava-v1.5-7b',
        method_id='unrelated-image-color',
        metrics_ids=['pearson', 'failure'],
        cfg=cfg,
    )
    
    task.pipeline()

def test_reproducible():
    task = ConfAIde_Task(
        task_id='confaide-task',
        dataset_id=dataset_ids[0],
        model_id='llava-v1.5-7b',
        metrics_ids=['pearson', 'failure'],
    )
    
    task.pipeline()

    # task = ConfAIde_Task(
    #     task_id='confaide-task',
    #     dataset_id=dataset_ids[1],
    #     model_id='llava-v1.5-7b',
    #     metrics_ids=['pearson', 'failure'],
    # )
    
    # task.pipeline()


if __name__ == '__main__':
    dataset_ids = ["confaide-text", "confaide-image", "confaide-unrelated-image-color", "confaide-unrelated-image-nature", "confaide-unrelated-image-noise"]
    
    # test_method()

    test_reproducible()
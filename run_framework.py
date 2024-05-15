from mmte.tasks import ConfAIde_Task

if __name__ == '__main__':
    dataset_ids = ["confaide-text", "confaide-image", "confaide-unrelated-image-color", "confaide-unrelated-image-nature", "confaide-unrelated-image-noise"]
    
    task = ConfAIde_Task(
        task_id='confaide-task',
        dataset_id=dataset_ids[2],
        model_id='llava-v1.5-7b',
        method_id=None,
        metrics_id='pearson',
        # metrics_id='failure',
    )
    
    task.run_task_from_scratch()

        
from mmte.models import load_chatmodel
from mmte.perspectives import get_task, task_pool
import os
import time
import argparse

def main(task, model_id, output_dir):
    # Your script logic here
    print(f"Task: {task}")
    print(f"Models to test: {model_id}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    task_handler = get_task(task)
    print("="*15)
    print(f"Evaluating {model_id}...")
    

    for k in range(10):
        try:
            test_model = load_chatmodel(model_id)
            break
        except:
            time.sleep(3)
        
    result = task_handler.eval(test_model, os.path.join(output_dir, f"{task}_on_{model_id}.json"))
    print(f"Results of task {task} on {model_id}: {result}")
    print("="*15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control script with command line arguments.")
    parser.add_argument("--task", type=str, required=True, help="Identifier of the task")
    parser.add_argument("--model_id", type=str, required=True, help="Identifier of the model")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save logs")

    args = parser.parse_args()
    print(args.task)

    jsonfile = os.path.join(args.output_dir, f"{args.task}_on_{args.model_id}.json")
    if os.path.exists(jsonfile):
        print(f"{jsonfile} already done!")
        exit(0)
    
    for task in task_pool():
        if args.task in task:
            main(task, args.model_id, args.output_dir)

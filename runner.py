import os
import subprocess
import argparse
from multiprocessing import Queue, Process

TASK_PER_GPU = None

log_queue = Queue()


def _log(msg):
    log_queue.put(msg)


def _log_task():
    while True:
        msg = log_queue.get()
        print(msg)


def get_gpus():
    return list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))


def run_task(k: int, q: Queue):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(k)
    pid = os.getpid()

    _log('GPU={} PID={} init'.format(k, pid))

    while True:
        item = q.get()
        if item == None:
            q.put(None)
            break
        else:
            i, cmd = item
            try:
                modelname = cmd.split('model_id=')[1].split(' ')[0]
                taskname = cmd.split('.yaml')[0].split('/')[-1].strip()
                subtaskname = cmd.split('/')[-1].split('.')[0].strip()
                logname = '-'.join([modelname, taskname, subtaskname])
            except:
                logname = 'out-{}'.format(i)
            _log('GPU={} PID={} I={} CMD="{}"'.format(k, pid, i, cmd))
            os.makedirs(args.log_dir, exist_ok=True)
            with open('{}/{}.log'.format(args.log_dir, logname), 'ab') as out:
                print(cmd)
                p = subprocess.Popen(cmd,
                                     shell=True,
                                     stdin=None,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     env=env)
                for line in iter(p.stdout.readline, b''):
                    out.write(line)
                    out.flush()
                    line = line.decode().strip('\n')
                    prefix = 'GPU={} PID={} I={}'.format(k, pid, i)
                    _log('{:25s} {}'.format(prefix, line))
                if p.stdout:
                    p.stdout.close()
                if p.stderr:
                    p.stderr.close()
                p.wait()
                del p
            _log('GPU={} PID={} I={} CMD="{}" done'.format(k, pid, i, cmd))

    _log('GPU={} PID={} exit'.format(k, pid))


def main(args):
    task_queue = Queue()

    task = args.task_list
    TASK_PER_GPU = args.task_per_gpu

    with open(task, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('#'):
                continue
            logfile = line.split('log_file=')[1].strip()
            if os.path.exists(logfile):
                print("{} is already exists.".format(logfile))
                continue
            task_queue.put((i, line.strip()))

    task_queue.put(None)

    gpus = get_gpus()

    logger = Process(target=_log_task)
    logger.start()

    ps = []
    for gpu in gpus:
        for _ in range(TASK_PER_GPU):
            p = Process(target=run_task, args=(gpu, task_queue))
            p.start()
            ps.append(p)

    for p in ps:
        p.join()

    logger.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-list', type=str, required=True)
    parser.add_argument('--task-per-gpu', type=int, required=True)
    parser.add_argument('--log-dir', type=str, default='./runlogs')
    args = parser.parse_args()
    main(args)
import argparse
import os
from datetime import datetime

from dateutil import tz

log_dir = None
reg_log_dir = None
LOG_FOUT = None
inited = False


def setup_log(args):
    global LOG_FOUT, log_dir, inited, start_time, reg_log_dir
    if inited:
        return
    inited = True
    config = args.config.split('/')[-1].split('.')[0].replace('config_baseline', 'cb')
    model_config = args.model_config.split('/')[-1].split('.')[0]
    tz_sh = tz.gettz('Asia/Shanghai')
    now = datetime.now(tz=tz_sh)
    if (not os.path.exists("./tf_logs")):
        os.mkdir("./tf_logs")
    # dir = '{}-{}-{}'.format(config, model_config, now.strftime("%m%d-%H%M%S"))
    dir = '{}-{}'.format(config, model_config)
    log_dir = os.path.join("./tf_logs", dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    os.system('rm -r {}'.format(os.path.join("./tf_logs", 'latest')))
    os.system("cd tf_logs && ln -s {} {} && cd ..".format(dir, "latest"))

    start_time = now
    LOG_FOUT = open(os.path.join(log_dir, 'log_train.txt'), 'w')
    log_string('log dir: {}'.format(log_dir))
    reg_log_dir = os.path.join(log_dir, "registration")
    if not os.path.exists(reg_log_dir):
        os.makedirs(reg_log_dir)


def log_string(out_str, end='\n'):
    LOG_FOUT.write(out_str)
    LOG_FOUT.write(end)
    LOG_FOUT.flush()
    print(out_str, end=end, flush=True)


def log_silent(out_str, end='\n'):
    LOG_FOUT.write(out_str)
    LOG_FOUT.write(end)
    LOG_FOUT.flush()


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--model_config', type=str)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.set_defaults(debug=False)
parser.add_argument('--checkpoint', type=str, required=False, help='Trained model weights', default="")
parser.add_argument('--weights')
parser.add_argument('--log', action='store_true')
parser.add_argument('--visualize', dest='visualize', action='store_true')
parser.set_defaults(visualize=False)
args = parser.parse_known_args()[0]
setup_log(args)


def is_last_run_end(last_run_file):
    with open(last_run_file) as f:
        lines = f.readlines()
        for i in lines:
            if 'end' in i:
                return True
    return False


cuda_dev = os.environ.get('CUDA_VISIBLE_DEVICE')
if cuda_dev is None:
    cuda_dev = '0'
last_run = 'lastrun_{}'.format(cuda_dev)
last_run_file = last_run + '.log'
last_run_id = 1
# while os.path.exists(last_run_file) and not is_last_run_end(last_run_file):
#     last_run_file = last_run + str(last_run_id) + '.log'
#     last_run_id += 1
# with open(last_run_file, 'w') as f:
#     f.write(f'start:{start_time.strftime("%m%d-%H%M%S")}\n')
#     f.write(f'log_dir:{log_dir}\n')
#     for k,v in vars(args).items():
#         f.write(f'{k}:{v}\n')

# @atexit.register
# def end_last_run():
#     tz_sh = tz.gettz('Asia/Shanghai')
#     now = datetime.now(tz=tz_sh)
#     with open(last_run_file, 'a') as f:
#         f.write(f'end:{now.strftime("%m%d-%H%M%S")}\n')

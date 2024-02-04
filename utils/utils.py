import os
import yaml

import shutil
import datetime
import subprocess
import slackweb

from argparse import ArgumentParser
from attrdict import AttrDict

def result_folder(args, save_folder_path):
    exp_num = 0
    while os.path.exists(f"{save_folder_path}/exp{exp_num}") == True:
        exp_num = exp_num + 1
    result_folder = f"{save_folder_path}/exp{exp_num}"
    os.makedirs(result_folder, exist_ok=True)
    shutil.copy(f'{args.config}', result_folder)
    f = open(os.path.join(result_folder, 'memo.txt'), 'w')
    f.write(f'実験設定など\n{args.memo}')
    f.close()

    return result_folder

def argparser():
    parser = ArgumentParser()
    
    parser.add_argument('--train_path', default='../given_data/original/X_train.npy', type=str, help='配布された学習用データのパス')
    parser.add_argument('--label_path', default='/work/data/original/y_train.npy', type=str, help='y_trainのパス')
    parser.add_argument('--x_test_path', default='/work/data/original/X_test.npy', type=str, help='x_testのパス')
    parser.add_argument('--sample_submit_path', default='/work/data/original/sample_submit.csv', type=str, help='sample_submitのパス')
    
    parser.add_argument('--after_x_train_path', default='/work/data/after_process_original/after_process_X_train.npy', type=str, help='after_process_original_pathのパス')
    parser.add_argument('--eval_data_path', default='/work/data/result/process_test_data.npy', type=str, help='eval_data_pathのパス')
    
    parser.add_argument('--dice3_imgs_path', default='/work/data/processed_original/avg_dice3_imgs.npy', type=str, help='dice3_imgs_pathのパス')
    parser.add_argument('--dice3_labels_path', default='/work/data/processed_original/avg_dice3_labels.npy', type=str, help='dice3_labels_pathのパス')
    
    parser.add_argument('--all_processed_imgs_path', default='/work/data/processed_original/avg_all_processed_imgs.npy', type=str, help='all_processed_imgs_pathのパス')
    parser.add_argument('--all_processed_labels_path', default='/work/data/processed_original/avg_all_processed_labels.npy', type=str, help='all_processed_labels_pathのパス')
    
    parser.add_argument('--submit_labels_path', default='/work/data/result/submit3.csv', type=str, help='submit_labels_pathのパス')
    
    parser.add_argument('--pretraind_model_path', default='/work/data/model_folder/model5_20.pth', type=str, help='model_submitのパス')
    
    parser.add_argument("--config", type=str, default='/work/src/config/config.yaml')
    parser.add_argument("--memo", type=str)
    parser.add_argument("--traind_model", action='store_true')
    
    parser.add_argument("--h", type=str, default='なし', help='補足説明')
    
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M")
    dir_for_output = './data/result/' + current_time
    os.makedirs(dir_for_output, exist_ok=True)
    parser.add_argument('--result_datetime_folder', default='./data/result/' + current_time, help='result_folderのパス')
    
    #cpuを変えるときに使う
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    return args

def config(args):
    cfg_path = args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = AttrDict(cfg)
    return cfg

def get_current_branch(repository_dir='./') -> str:
    cmd = "cd %s && git rev-parse --abbrev-ref HEAD" % repository_dir
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    stdout_data = proc.stdout.read()
    current_branch = stdout_data.decode('utf-8').replace('\n','')
    return current_branch

args = argparser()
cfg = config(args)

if cfg.switch.slack == True:
    cmd = ["git", "push"] 
    subprocess.run(cmd)
    slack = slackweb.Slack(url='https://hooks.slack.com/services/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')
    push_branch = get_current_branch()
    text = f'{os.environ.get("USER")}さんがbranch:{push_branch}にpushしました.補足:{args.h}'
    slack.notify(text=text)
else:
    pass
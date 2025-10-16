import argparse
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from experiments.exp import Exp
import optuna
from optuna.samplers import TPESampler
from datetime import datetime
import csv


parser = argparse.ArgumentParser(description='SCINet on ETT dataset')


parser.add_argument('--model', type=str, required=False, default='SCINet', help='model of the experiment')
### -------  dataset settings --------------
parser.add_argument('--data', type=str, required=False, default='weather', choices=['ETTh2', 'ETTm1', 'ETTm2'], help='name of dataset')
parser.add_argument('--root_path', type=str, default='/root/autodl-tmp/SCINetmainWithTFEDNC3 copy/datasets/weather/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='weather.csv', help='location of the data file')
parser.add_argument('--features', type=str, default='M', choices=['S', 'M'], help='features S is univariate, M is multivariate')
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='/root/autodl-tmp/SCINetmainWithTFEDNC3 copy/exp/weather_checkpoints/weather_result', help='location of model checkpoints')
parser.add_argument('--inverse', type=bool, default =False, help='denorm the output data')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')


### -------  device settings --------------
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0',help='device ids of multile gpus')
                                                                                  
### -------  input/output length settings --------------                                                                            
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of SCINet encoder, look back window')  #ori_48pred_96 new_48pred_128
parser.add_argument('--label_len', type=int, default=0, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length, horizon')
parser.add_argument('--concat_len', type=int, default=0)
parser.add_argument('--single_step', type=int, default=0)
parser.add_argument('--single_step_output_One', type=int, default=0)
parser.add_argument('--lastWeight', type=float, default=1.0)
                                                              
### -------  training settings --------------  
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=0, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=720, help='batch size of train input data')  ### 32
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--lr', type=float, default=0.01, help='optimizer learning rate') ### 0.0001-ETT
parser.add_argument('--loss', type=str, default='mae',help='loss function')
parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--save', type=bool, default =False, help='save the output results')
parser.add_argument('--model_name', type=str, default='SCINet-ADFNet')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--evaluate', type=bool, default=False)

parser.add_argument('--hidden-size', default=4, type=float, help='hidden channel of module') ##  1
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=5, type=int, help='kernel size, 3, 5, 7')
parser.add_argument('--dilation', default=3, type=int, help='dilation') #1
parser.add_argument('--window_size', default=36, type=int, help='input size')  ## 12
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')  # 0.5
parser.add_argument('--positionalEcoding', type=bool, default=True)  # False 默认为False
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--levels', type=int, default=3) # 3
parser.add_argument('--stacks', type=int, default=1, help='1 stack or 2 stacks')## 1 
parser.add_argument('--num_decoder_layer', type=int, default=1)  # 1
parser.add_argument('--RIN', type=bool, default=False)  # False 默认为False

#     +++++++++++新增机制+++++++     #


parser.add_argument('--fft_ratio', type=float, default=0.65, help='the component of SCINet/use_fft = True')  # 可以调整数值，default 0.55  0.8407898229562142
parser.add_argument('--use_fft', action='store_true', default=False, help='the component of SCINet')  
parser.add_argument('--fft_mode', type=str, default='combined', help='split or combined/the component of SCINet/use_fft = True')   # 有split和combined两种，详见SCINet.py里面的Interactor部分  combined
parser.add_argument('--middle_layer_count', type=int, default=0, help='[1, 2, 3]') # 0
parser.add_argument('--activation', default='Tanh', help='选择激活函数，可选项: GELU, SELU, Softplus, ELU, Tanh') # GELU
parser.add_argument('--dropout1d_v', type=float, default=0.15, help='[from 0.1 to 0.6]')  # 0.5


# 新增参数解析
parser.add_argument('--hpo', action='store_true', default = False, help='启用超参数优化')

parser.add_argument('--tune-lr', action='store_true', default = True, help='调优学习率')
parser.add_argument('--tune-bs', action='store_true', default = True, help='调优batch_size')
parser.add_argument('--tune-dp', action='store_true', default = True, help='调优dropout')
parser.add_argument('--tune-hs', action='store_true', default = True, help='调优hidden_size')
parser.add_argument('--tune-dil', action='store_true', default = True, help='调优dilation')
parser.add_argument('--tune-win', action='store_true', default = True, help='调优window_size')

parser.add_argument('--hpo-trials', type=int, default=30, help='Optuna试验次数')

parser.add_argument('--seed', type=int, default=4321, help='全局随机种子')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
    'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
    'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

torch.manual_seed(4321)  # reproducible
torch.cuda.manual_seed_all(4321)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
torch.backends.cudnn.enabled = True

Exp = Exp

mae_ = []
maes_ = []
mse_ = []
mses_ = []

if args.hpo:
    print(">>>>>>> Starting hyperparameter optimization <<<<<<<<")

    # 创建结果目录
    result_dir = os.path.join(args.checkpoints, 'hpo_results')
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f'hpo_{datetime.now().strftime("%Y%m%d%H%M")}.csv')

    def objective(trial):
        # 克隆基础参数
        trial_args = argparse.Namespace(**vars(args))
        
        # 定义参数搜索空间
        param_space = {
            'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True) if args.tune_lr else args.lr,
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]) if args.tune_bs else args.batch_size,
            'dropout': trial.suggest_float('dropout', 0.1, 0.8) if args.tune_dp else args.dropout,
            'hidden-size': trial.suggest_int('hidden-size', 1, 6, step=1) if args.tune_hs else args.hidden_size,  ####
            'dilation': trial.suggest_int('dilation', 1, 3) if args.tune_dil else args.dilation,
            'window_size': trial.suggest_categorical('window_size', [12, 24, 36]) if args.tune_win else args.window_size,
           
            'positionalEcoding': trial.suggest_categorical('positionalEcoding', [True, False]),
            'RIN': trial.suggest_categorical('RIN', [True, False])
        }

        # === FFT参数处理 ===
        if trial_args.use_fft:
            param_space.update({
                'fft_ratio': trial.suggest_float('fft_ratio', 0.1, 0.9),
                'fft_mode': trial.suggest_categorical('fft_mode', ['split', 'combined']),
                'middle_layer_count': trial.suggest_int('middle_layer_count', 1, 3),
                'activation': trial.suggest_categorical('activation', ['GELU', 'SELU', 'Softplus', 'ELU', 'Tanh']),
                'dropout1d_v': trial.suggest_float('dropout1d_v', 0.1, 0.7)
            })

        # 注入参数到试验配置
        for k, v in param_space.items():
            setattr(trial_args, k, v)

        # 优化训练配置
        trial_args.patience = 5  # 早停等待 5
        trial_args.save = False   # 不保存中间模型

        try:
            # 训练验证流程

            # ===== 生成唯一实验标识 =====
            setting = f"hpo_trial_{trial.number}_{datetime.now().strftime('%m%d%H%M')}"
        
            exp = Exp(trial_args)
            exp.train(setting) # 正常训练，返回模型
            
            val_mae = exp.best_val_mae  # 通过新增属性获得

            return val_mae  # 返回MAE作为优化目标

        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            return float('inf')

    # 配置优化器
    sampler = TPESampler(seed=args.seed)  # 可重复采样
    study = optuna.create_study(
        direction='minimize',  # 验证MAE越小越好
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner()
    )

    # 执行优化
    study.optimize(objective, n_trials=args.hpo_trials, show_progress_bar=True)

    # 输出最佳结果
    print("\n>>>>>>> Optimization results <<<<<<<<")
    print(f"Best normed mae: {study.best_value:.4f}")
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 用最佳参数进行最终训练
    print("\n>>>>>>> Final training with best parameters <<<<<<<<")
    best_args = argparse.Namespace(**vars(args))
    for k, v in study.best_params.items():
        setattr(best_args, k, v)
    
    final_exp = Exp_ETTh(best_args)
    final_exp.train('best_config', return_best_val_mae=True)
    
   
elif args.evaluate:
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_hid{}_s{}_l{}_dp{}_inv{}_itr0'.format(args.model,args.data, args.features, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size,args.hidden_size,args.stacks, args.levels,args.dropout,args.inverse)
    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae, maes, mse, mses = exp.test(setting, evaluate=True)
    print('Final mean normed mse:{:.4f},mae:{:.4f},denormed mse:{:.4f},mae:{:.4f}'.format(mse, mae, mses, maes))
else:
    if args.itr:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_hid{}_s{}_l{}_dp{}_inv{}_itr{}'.format(args.model,args.data, args.features, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size,args.hidden_size,args.stacks, args.levels,args.dropout,args.inverse,ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mae, maes, mse, mses = exp.test(setting)
            mae_.append(mae)
            mse_.append(mse)
            maes_.append(maes)
            mses_.append(mses)

            torch.cuda.empty_cache()
        print('Final mean normed mse:{:.4f}, std mse:{:.4f}, mae:{:.4f}, std mae:{:.4f}'.format(np.mean(mse_), np.std(mse_), np.mean(mae_),np.std(mae_)))
        print('Final mean denormed mse:{:.4f}, std mse:{:.4f}, mae:{:.4f}, std mae:{:.4f}'.format(np.mean(mses_),np.std(mses_), np.mean(maes_), np.std(maes_)))
        print('Final min normed mse:{:.4f}, mae:{:.4f}'.format(min(mse_), min(mae_)))
        print('Final min denormed mse:{:.4f}, mae:{:.4f}'.format(min(mses_), min(maes_)))
    else:
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_hid{}_s{}_l{}_dp{}_inv{}_itr0'.format(args.model,args.data, args.features, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size,args.hidden_size,args.stacks, args.levels,args.dropout,args.inverse)
        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mae, maes, mse, mses = exp.test(setting)
        print('Final mean normed mse:{:.4f},mae:{:.4f},denormed mse:{:.4f},mae:{:.4f}'.format(mse, mae, mses, maes))




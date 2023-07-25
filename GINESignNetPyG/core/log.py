# Create a simple logger that can log training curves and final performance 
from torch.utils.tensorboard import SummaryWriter # tensorboard
import logging, os, sys, shutil
import datetime

def config_logger(cfg, OUT_PATH="results/", time=False):
    # time option is used for debugging different model architecture. 
    data_name = cfg.dataset 
    if cfg.handtune:
        data_name += f'-{cfg.handtune}'
    # generate config_string
    os.makedirs(os.path.join(OUT_PATH, cfg.version), exist_ok=True)
    config_string = f'T[{cfg.task}] GNN[{cfg.model.gnn_type}] L[{cfg.model.num_layers_sign}-{cfg.model.num_layers}]  '\
                    f'H[{cfg.model.hidden_size}] Pool[{cfg.model.pool}] '\
                    f'Reg[{cfg.train.dropout}-{cfg.train.wd}] Seed[{cfg.seed}] GPU[{cfg.device}]'
    
    # setup tensorboard writer
    writer_folder = os.path.join(OUT_PATH, cfg.version, data_name, config_string)
    if time:
        writer_folder = os.path.join(writer_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if os.path.isdir(writer_folder): shutil.rmtree(writer_folder) # reset the folder, can also not reset
    writer = SummaryWriter(writer_folder)

    # setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger_filer = os.path.join(OUT_PATH, cfg.version, data_name, 'summary.log')
    fh = logging.FileHandler(logger_filer)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

    # redirect stdout print, better for large scale experiments
    os.makedirs(os.path.join('logs', data_name), exist_ok=True)
    # sys.stdout = open(f'logs/{data_name}/{config_string}.txt', 'w')

    # log configuration 
    print("-"*50)
    print(cfg)
    print("-"*50)
    print('Time:', datetime.datetime.now().strftime("%Y/%m/%d - %H:%M"))
    print(config_string)
    return writer, logger, config_string
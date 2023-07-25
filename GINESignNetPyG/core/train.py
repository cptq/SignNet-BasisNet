import torch
import time
from core.log import config_logger
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR

def run(cfg, create_dataset, create_model, train, test, evaluator=None):
    # set num threads
    # torch.set_num_threads(cfg.num_workers)

    # 0. create logger and writer
    writer, logger, config_string = config_logger(cfg)

    # 1. create dataset
    train_dataset, val_dataset, test_dataset = create_dataset(cfg)

    # 2. create loader
    train_loader = DataLoader(train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset,  cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    test_perfs = []
    vali_perfs = []
    for run in range(1, cfg.train.runs+1):
        # 3. create model and opt
        model = create_model(cfg).to(cfg.device)
        # print(f"Number of parameters: {count_parameters(model)}")
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        scheduler = StepLR(optimizer, step_size=cfg.train.lr_patience, gamma=cfg.train.lr_decay)

        # 4. train
        start_outer = time.time()
        best_val_perf = test_perf = float('-inf')
        for epoch in range(1, cfg.train.epochs+1):
            start = time.time()
            model.train()
            train_loss = train(train_loader, model, optimizer, device=cfg.device)
            scheduler.step()
            memory_allocated = torch.cuda.max_memory_allocated(cfg.device) // (1024 ** 2)
            memory_reserved = torch.cuda.max_memory_reserved(cfg.device) // (1024 ** 2)
            # print(f"---{test(train_loader, model, evaluator=evaluator, device=cfg.device) }")

            model.eval()
            val_perf = test(val_loader, model, evaluator=evaluator, device=cfg.device)
            if val_perf > best_val_perf:
                best_val_perf = val_perf
                test_perf = test(test_loader, model, evaluator=evaluator, device=cfg.device) 
            time_per_epoch = time.time() - start 

            # logger here
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                  f'Val: {val_perf:.4f}, Test: {test_perf:.4f}, Seconds: {time_per_epoch:.4f}, '
                  f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')

            # logging training
            writer.add_scalar(f'Run{run}/train-loss', train_loss, epoch)
            writer.add_scalar(f'Run{run}/val-perf', val_perf, epoch)
            writer.add_scalar(f'Run{run}/test-best-perf', test_perf, epoch)
            writer.add_scalar(f'Run{run}/seconds', time_per_epoch, epoch)   
            writer.add_scalar(f'Run{run}/memory', memory_allocated, epoch)   

            torch.cuda.empty_cache() # empty test part memory cost

        time_average_epoch = time.time() - start_outer
        print(f'Run {run}, Vali: {best_val_perf}, Test: {test_perf}, Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        test_perfs.append(test_perf)
        vali_perfs.append(best_val_perf)

    test_perf = torch.tensor(test_perfs)
    vali_perf = torch.tensor(vali_perfs)
    logger.info("-"*50)
    logger.info(config_string)
    # logger.info(cfg)
    logger.info(f'Final Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}, Final Test: {test_perf.mean():.4f} ± {test_perf.std():.4f},'
                f'Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
    print(f'Final Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}, Final Test: {test_perf.mean():.4f} ± {test_perf.std():.4f},'
                f'Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')


def run_k_fold(cfg, create_dataset, create_model, train, test, evaluator=None, k=10):
    # if cfg.seed is not None:

    writer, logger, config_string = config_logger(cfg)
    dataset, transform, transform_eval = create_dataset(cfg)

    if hasattr(dataset, 'train_indices'):
        k_fold_indices = dataset.train_indices, dataset.test_indices
    else:
        k_fold_indices = k_fold(dataset, k)

    test_perfs = []
    test_curves = []
    for fold, (train_idx, test_idx) in enumerate(zip(*k_fold_indices)):
        set_random_seed(0) # important 
        
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        train_dataset.transform = transform
        test_dataset.transform = transform_eval
        test_dataset = [x for x in test_dataset]
        train_dataset = [x for x in train_dataset]

        train_loader = DataLoader(train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
        test_loader = DataLoader(test_dataset,  cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)

        model = create_model(cfg).to(cfg.device)
        model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        scheduler = StepLR(optimizer, step_size=cfg.train.lr_patience, gamma=cfg.train.lr_decay)

        start_outer = time.time()
        best_test_perf = test_perf = float('-inf')
        test_curve = []
        for epoch in range(1, cfg.train.epochs+1):
            start = time.time()
            model.train()
            train_loss = train(train_loader, model, optimizer, device=cfg.device)
            scheduler.step()
            memory_allocated = torch.cuda.max_memory_allocated(cfg.device) // (1024 ** 2)
            memory_reserved = torch.cuda.max_memory_reserved(cfg.device) // (1024 ** 2)

            model.eval()
            test_perf = test(test_loader, model, evaluator=evaluator, device=cfg.device) 
            test_curve.append(test_perf.item())
            best_test_perf = test_perf if test_perf > best_test_perf else best_test_perf
  
            time_per_epoch = time.time() - start 

            # logger here
            print(f'Epoch/Fold: {epoch:03d}/{fold}, Train Loss: {train_loss:.4f}, '
                  f'Test:{test_perf:.4f}, Best-Test: {best_test_perf:.4f}, Seconds: {time_per_epoch:.4f}, '
                  f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')

            # logging training
            writer.add_scalar(f'Fold{fold}/train-loss', train_loss, epoch)
            writer.add_scalar(f'Fold{fold}/test-perf', test_perf, epoch)
            writer.add_scalar(f'Fold{fold}/test-best-perf', best_test_perf, epoch)
            writer.add_scalar(f'Fold{fold}/seconds', time_per_epoch, epoch)   
            writer.add_scalar(f'Fold{fold}/memory', memory_allocated, epoch)   

            torch.cuda.empty_cache() # empty test part memory cost

        time_average_epoch = time.time() - start_outer
        print(f'Fold {fold}, Test: {best_test_perf}, Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        test_perfs.append(best_test_perf)
        test_curves.append(test_curve)

    logger.info("-"*50)
    logger.info(config_string)
    test_perf = torch.tensor(test_perfs)
    logger.info(" ===== Final result 1, based on average of max validation  ========")
    print(" ===== Final result 1, based on average of max validation  ========")
    msg = (
        f'Dataset:        {cfg.dataset}\n'
        f'Accuracy:       {test_perf.mean():.4f} ± {test_perf.std():.4f}\n'
        f'Seconds/epoch:  {time_average_epoch/cfg.train.epochs}\n'
        f'Memory Peak:    {memory_allocated} MB allocated, {memory_reserved} MB reserved.\n'
        '-------------------------------\n')
    logger.info(msg)
    print(msg)  

    logger.info("-"*50)
    test_curves = torch.tensor(test_curves)
    avg_test_curve = test_curves.mean(axis=0)
    best_index = np.argmax(avg_test_curve)
    mean_perf = avg_test_curve[best_index]
    std_perf = test_curves.std(axis=0)[best_index]

    logger.info(" ===== Final result 2, based on average of validation curve ========")
    print(" ===== Final result 2, based on average of validation curve ========")
    msg = (
        f'Dataset:        {cfg.dataset}\n'
        f'Accuracy:       {mean_perf:.4f} ± {std_perf:.4f}\n'
        f'Best epoch:     {best_index}\n'
        '-------------------------------\n')
    logger.info(msg)
    print(msg)   

import random, numpy as np
import warnings
def set_random_seed(seed=0, cuda_deterministic=True):
    """
    This function is only used for reproducbility, 
    DDP model doesn't need to use same seed for model initialization, 
    as it will automatically send the initialized model from master node to other nodes. 
    Notice this requires no change of model after call DDP(model)
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training with CUDNN deterministic setting,'
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        warnings.warn('You have chosen to seed training WITHOUT CUDNN deterministic. '
                       'This is much faster but less reproducible')

from sklearn.model_selection import StratifiedKFold
def k_fold(dataset, folds=10):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    train_indices, test_indices = [], []
    ys = dataset.data.y
    # ys = [graph.y.item() for graph in dataset]
    for train, test in skf.split(torch.zeros(len(dataset)), ys):
        train_indices.append(torch.from_numpy(train).to(torch.long))
        test_indices.append(torch.from_numpy(test).to(torch.long))
        # train_indices.append(train)
        # test_indices.append(test)
    return train_indices, test_indices

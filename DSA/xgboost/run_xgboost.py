import sys
from pathlib import Path
prj_path = Path(__file__).parent.resolve().parent.resolve()
sys.path.append(str(prj_path))
import argparse
import os
import random


import json, bz2, pickle
import numpy as np
import pandas as pd

import multiprocessing
from functools import partial
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
import copy

import optuna
import logging
from io import StringIO
import time

from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler, SMOTE, KMeansSMOTE, ADASYN


def freeze_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    """
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    """

class FoldLoggerStream(StringIO):
    def __init__(self, logger, level=logging.INFO):
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, buf):
        output = buf.strip()
        if output:
            self.logger.log(self.level, output)

def set_foldlogger(args, k):
    if args.name_only:
        log_path = Path(str(args.prj_path)+'/logs').joinpath(Path(str(args.optuna_log_path)+'/nameonly'))
    elif args.name_erased:
        log_path = Path(str(args.prj_path)+'/logs').joinpath(Path(str(args.optuna_log_path)+'/nameerased'))
    else:
        log_path = Path(str(args.prj_path)+'/logs').joinpath(Path(str(args.optuna_log_path)+'/fulldescription'))
    log_path.mkdir(parents=True, exist_ok=True)
    log_handler = logging.FileHandler(log_path / f"fold{k}.log", mode="w")
    fold_logger = logging.getLogger(f"fold{k}")
    fold_logger.addHandler(log_handler)

    return fold_logger

def load_and_process_seqfea(path, name):
    return np.mean(np.load(path / name / f'{name}.npy'), axis=-2)

def standard_scale(data_tuple):
    n = 0  # total number of samples seen so far
    mean = 0  # total mean
    M2 = 0  # median value of (x - mean)^2
    for array in data_tuple:
        n += array.shape[0]
        delta = array - mean
        mean += np.sum(delta, axis=0) / n
        M2 += np.sum(delta * (array - mean), axis=0)
    variance = M2 / n
    std = np.sqrt(variance)
    return mean, std

def init_folddata(args, k):
    if args.optuna_log_path.split('_')[-1] == 'NC':
        # load data
        train_data = pd.read_csv(Path(str(args.label_path)) / str(f'fold{k}_'+args.optuna_log_path.split('_')[-1]) / 'train.csv')
        valid_data = pd.read_csv(Path(str(args.label_path)) / str(f'fold{k}_'+args.optuna_log_path.split('_')[-1]) / 'dev.csv')
        print('data loaded')
        if args.negtive_sample == 'equal':
            train_data = train_data[train_data['label'] == 0].sample(n=len(train_data[train_data['label'] != 0])).append(train_data[train_data['label'] != 0])
        elif args.negtive_sample == 'none':
            train_data = train_data[train_data['label'] != 0]
        elif args.negtive_sample == 'all':
            pass

        # load embedding
        if args.name_only:
            compoundfea_path = Path(str(args.feature_path)+f'/llama_{args.llama_size}') / str('Drug_'+args.optuna_log_path.split('_')[-1].lower()+'_nameonly')
            diseasefea_path = Path(str(args.feature_path)+f'/llama_{args.llama_size}') / str('Sideeffect_'+args.optuna_log_path.split('_')[-1].lower()+'_nameonly')
        elif args.name_erased:
            compoundfea_path = Path(str(args.feature_path)+f'/llama_{args.llama_size}') / str('Drug_'+args.optuna_log_path.split('_')[-1].lower()+'_nameerased')
            diseasefea_path = Path(str(args.feature_path)+f'/llama_{args.llama_size}') / str('Sideeffect_'+args.optuna_log_path.split('_')[-1].lower()+'_nameerased')
        else:
            compoundfea_path = Path(str(args.feature_path)+f'/llama_{args.llama_size}') / str('Drug_'+args.optuna_log_path.split('_')[-1].lower())
            diseasefea_path = Path(str(args.feature_path)+f'/llama_{args.llama_size}') / str('Sideeffect_'+args.optuna_log_path.split('_')[-1].lower())

        # train
        pool = multiprocessing.Pool(processes=args.num_works)
        train_src = pool.map(partial(load_and_process_seqfea,compoundfea_path),train_data['drug'])
        train_tar = pool.map(partial(load_and_process_seqfea,diseasefea_path),train_data['sideeffect'])
        pool.close()
        pool.join()
        train = np.concatenate([np.array(train_src,dtype=np.float32), np.array(train_tar,dtype=np.float32)], axis=1)
        train_label = np.array(train_data['label'], dtype=np.int64)
        # valid
        pool = multiprocessing.Pool(processes=args.num_works)
        valid_src = pool.map(partial(load_and_process_seqfea,compoundfea_path),valid_data['drug'])
        valid_tar = pool.map(partial(load_and_process_seqfea,diseasefea_path),valid_data['sideeffect'])
        pool.close()
        pool.join()
        valid = np.concatenate([np.array(valid_src,dtype=np.float32), np.array(valid_tar,dtype=np.float32)], axis=1)
        valid_label = np.array(valid_data['label'], dtype=np.int64)

    index = np.arange(train_label.shape[0])
    np.random.shuffle(index)
    train, train_label = train[index].squeeze(), train_label[index].squeeze()

    mean, std = standard_scale((train, valid))
    train = (train - mean) / std
    valid = (valid - mean) / std

    print("mean.shape, std.shape: ", mean.shape, std.shape)
    print("mean & std saved to: ", str(Path(str(args.label_path))/str(f'fold{k}_'+args.optuna_log_path.split('_')[-1])/'mean.npy'), ' & ' , str(Path(str(args.label_path))/str(f'fold{k}_'+args.optuna_log_path.split('_')[-1])/'std.npy'))
    np.save(Path(str(args.label_path)) / str(f'fold{k}_'+args.optuna_log_path.split('_')[-1]) / 'mean.npy', mean)
    np.save(Path(str(args.label_path)) / str(f'fold{k}_'+args.optuna_log_path.split('_')[-1]) / 'std.npy', std)
    print('data inited')
    print(f'inited data info:')
    print(f'train: {train.shape}, valid: {valid.shape}')
    print(f"memory used: train={train.nbytes/(1024**3)} GB, valid={valid.nbytes/(1024**3)} GB, ")
    return (train, train_label), (valid, valid_label)

def superIO(args, results, fold, optuna_hpps):
    print('saving results while saveflag activated...')

    if args.name_only:
        saveio_path = Path(str(args.prj_path)+'/pretrained').joinpath(Path(str(args.pretrained_data_path)+'/nameonly'))
    elif args.name_erased:
        saveio_path = Path(str(args.prj_path)+'/pretrained').joinpath(Path(str(args.pretrained_data_path)+'/nameerased'))
    else: 
        saveio_path = Path(str(args.prj_path)+'/pretrained').joinpath(Path(str(args.pretrained_data_path)+'/fulldescription'))
    saveio_path.mkdir(parents=True, exist_ok=True)

    def save_model(args, fold, model, optuna_hpps):
        model_savepath = saveio_path / 'pretrained_model' / f'{fold}th_Fold'
        model_savepath.mkdir(parents=True, exist_ok=True)
        with open(model_savepath / f'model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open(model_savepath / f'hpps.json', 'w') as f:
            json.dump(optuna_hpps, f)

    def save_curves(args, k, xs_dict, ys_dict, thresholds_dict, flag='train_ROC'):
        # save best kth Fold ROC data
        curve_savepath = saveio_path / f'{flag}_data' / f'{k}th_Fold'
        curve_savepath.mkdir(parents=True, exist_ok=True)
        for etk in xs_dict:
            xs, ys, thresholds = xs_dict[etk], ys_dict[etk], thresholds_dict[etk]
            pd.DataFrame({'fprs':xs, 'tprs':ys, 'thresholds':thresholds}).to_csv(curve_savepath / f'{flag}_of_{etk}_for_{k}th_Fold.csv')

    def save_metrics(args, fold, results):
        metric_savepath = saveio_path / 'metric_data' / f'{fold}th_Fold'
        metric_savepath.mkdir(parents=True, exist_ok=True)
        for etk in results[1][6]:
            pd.DataFrame({'auc':results[1][6][etk], 'tn':results[1][7][etk], 'fp':results[1][8][etk], 'fn':results[1][9][etk], 'tp':results[1][10][etk], 'acc':results[1][11][etk], 'mcc':results[1][12][etk], 'precision':results[1][13][etk], 'recall':results[1][14][etk], 'specificity':results[1][15][etk], 'sensitivity':results[1][16][etk], 'f1':results[1][17][etk], 'prauc':results[1][18][etk], 'av_prc':results[1][19][etk], 'macro_f1':results[1][20][etk], 'kappa':results[1][21][etk], 'rmse':results[1][22][etk], 'mae':results[1][23][etk], 'pcc':results[1][24][etk]}, index=[etk]).to_csv(metric_savepath / f'train_metrics_of_{etk}_for_{fold}th_Fold.csv')
            pd.DataFrame({'auc':results[2][6][etk], 'tn':results[2][7][etk], 'fp':results[2][8][etk], 'fn':results[2][9][etk], 'tp':results[2][10][etk], 'acc':results[2][11][etk], 'mcc':results[2][12][etk], 'precision':results[2][13][etk], 'recall':results[2][14][etk], 'specificity':results[2][15][etk], 'sensitivity':results[2][16][etk], 'f1':results[2][17][etk], 'prauc':results[2][18][etk], 'av_prc':results[2][19][etk], 'macro_f1':results[2][20][etk], 'kappa':results[2][21][etk], 'rmse':results[2][22][etk], 'mae':results[2][23][etk], 'pcc':results[2][24][etk]}, index=[etk]).to_csv(metric_savepath / f'valid_metrics_of_{etk}_for_{fold}th_Fold.csv')

    def save_logits(args, fold, results):
        logit_savepath = saveio_path / 'logit_data' / f'{fold}th_Fold'
        logit_savepath.mkdir(parents=True, exist_ok=True)
        for etk in results[4][0]:
            pd.DataFrame({'y_true_train_logits':results[4][0][etk][0], 'y_scores_train_logits':[str(i) for i in results[4][0][etk][1]], 'threshold_train_logits':np.array([results[4][0][etk][2] for i in range(len(results[4][0][etk][0]))])}).to_csv(logit_savepath / f'train_logits_of_{etk}_for_{fold}th_Fold.csv')
            pd.DataFrame({'y_true_valid_logits':results[4][1][etk][0], 'y_scores_valid_logits':[str(i) for i in results[4][1][etk][1]], 'threshold_valid_logits':np.array([results[4][1][etk][2] for i in range(len(results[4][1][etk][0]))])}).to_csv(logit_savepath / f'valid_logits_of_{etk}_for_{fold}th_Fold.csv')

    # save best kth model
    save_model(args, fold, results[0][0], optuna_hpps)
    # save best kth Fold ROC data
    save_curves(args, fold, results[1][0], results[1][1], results[1][2], flag='train_ROC')
    save_curves(args, fold, results[2][0], results[2][1], results[2][2], flag='valid_ROC')

    # save best kth Fold PRC data
    save_curves(args, fold, results[1][3], results[1][4], results[1][5], flag='train_PRC')
    save_curves(args, fold, results[2][3], results[2][4], results[2][5], flag='valid_PRC')

    # save best kth Fold metrics
    save_metrics(args, fold, results)

    # save best kth Fold logits
    save_logits(args, fold, results)

def calculate_fusion_scores(y_true, y_scores, threshold=0.5):

    fprs, tprs, thresholds, pres, recs, thresholds_prc, auc, tn, fp, fn, tp, acc, mcc, precision, recall, specificity, sensitivity, f1, prauc, av_prc, macro_f1, kappa, rmse, mae, pcc = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if len(y_true) == 0 or len(y_scores) == 0:
        pass

    elif len(np.unique(y_true)) == 2:
        y_scores = y_scores[:,1]
        # ROC, AUC
        fprs, tprs, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)
        auc = metrics.auc(fprs, tprs)
        # prauc
        pres, recs, thresholds_prc = metrics.precision_recall_curve(y_true, y_scores, pos_label=1)
        thresholds_prc = np.append(thresholds_prc, [1.0], axis=0)
        prauc = metrics.auc(recs, pres)
        av_prc = metrics.average_precision_score(y_true, y_scores)
        # scores' label prediction by threshold
        label_pred = copy.deepcopy(y_scores)
        label_pred = np.where(y_scores >= threshold, np.ones_like(label_pred), label_pred)
        label_pred = np.where(y_scores < threshold, np.zeros_like(label_pred), label_pred)
        # TN, FP, FN, TP
        tn, fp, fn, tp = metrics.confusion_matrix(y_true=y_true, y_pred=label_pred, labels=[0,1]).ravel()
        # Model Evaluation
        acc = metrics.accuracy_score(y_true, label_pred)
        mcc = metrics.matthews_corrcoef(y_true, label_pred)
        precision = metrics.precision_score(y_true, label_pred)
        specificity = tn/(tn+fp)
        sensitivity = tp/(tp+fn)
        recall = metrics.recall_score(y_true, label_pred)
        f1 = metrics.f1_score(y_true, label_pred)

    elif len(y_scores.shape) == 1:
        # calculate RMSE
        rmse = np.sqrt(metrics.mean_squared_error(y_true[np.nonzero(y_true)], y_scores[np.nonzero(y_true)]))
        # calculate MAE
        mae = metrics.mean_absolute_error(y_true[np.nonzero(y_true)], y_scores[np.nonzero(y_true)])
        # calculate PCC
        pcc = np.corrcoef(y_true[np.nonzero(y_true)], y_scores[np.nonzero(y_true)])[0,1]

        # ROC, AUC
        fprs, tprs, thresholds = metrics.roc_curve(np.where(y_true != 0, 1, y_true), y_scores)
        auc = metrics.auc(fprs, tprs)
        # prauc
        pres, recs, thresholds_prc = metrics.precision_recall_curve(np.where(y_true != 0, 1, y_true), y_scores)
        thresholds_prc = np.append(thresholds_prc, [1.0], axis=0)
        prauc = metrics.auc(recs, pres)

    else: 
        y_pred = np.argmax(y_scores, axis=1)
        macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
        acc = metrics.accuracy_score(y_true, y_pred)
        kappa = metrics.cohen_kappa_score(y_true, y_pred)
        
        n_classes = y_scores.shape[1]
        # macro AUROC, AUPRC
        # auc = metrics.roc_auc_score(y_true, y_scores, average='macro', multi_class='ovo')
        fprs_dicttmp, tprs_dicttmp, aucs_dicttmp = dict(),dict(),dict()
        precisions_dicttmp,recalls_dicttmp,praucs_dicttmp = dict(),dict(),dict()
        for i in range(n_classes):
            fprs_dicttmp[i], tprs_dicttmp[i], _ = metrics.roc_curve(y_true == i, y_scores[:, i])
            aucs_dicttmp[i] = metrics.auc(fprs_dicttmp[i], tprs_dicttmp[i])
            precisions_dicttmp[i], recalls_dicttmp[i], _ = metrics.precision_recall_curve(y_true == i, y_scores[:, i])
            praucs_dicttmp[i] = metrics.auc(recalls_dicttmp[i], precisions_dicttmp[i])
        auc = sum(aucs_dicttmp.values()) / n_classes
        prauc = sum(praucs_dicttmp.values()) / n_classes

        precision = metrics.precision_score(y_true, y_pred, zero_division=0, average='macro')
        recall = metrics.recall_score(y_true, y_pred, zero_division=0, average='macro')

    return fprs, tprs, thresholds, pres, recs, thresholds_prc, auc, tn, fp, fn, tp, acc, mcc, precision, recall, specificity, sensitivity, f1, prauc, av_prc, macro_f1, kappa, rmse, mae, pcc

def evaluate_xgb(train, valid, num_workers, optuna_hpps):
    # create model instance

    bst = XGBRegressor(n_jobs=num_workers,booster='gbtree',eval_metric ='rmse',objective='reg:squarederror',tree_method = 'auto',seed=42,**optuna_hpps)
    # fit model
    unique_values, counts = np.unique(train[1], return_counts=True)
    proportions = [np.sum(counts)/count for count in counts]
    replace_dict = {unique_values[i]: proportions[i]/np.sum(proportions) for i in range(len(unique_values))}

    print("replace_dict: ", replace_dict)
    bst.fit(train[0], train[1], sample_weight=np.vectorize(lambda x: replace_dict.get(x, x))(train[1]))
    
    train_preds = bst.predict(train[0])
    valid_preds = bst.predict(valid[0])

    threshold_tr = 0.5
    threshold_v = 0.5

    return bst, (train[1], train_preds, threshold_tr), (valid[1], valid_preds, threshold_v), calculate_fusion_scores(train[1], train_preds, threshold_tr), calculate_fusion_scores(valid[1], valid_preds, threshold_v)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random_seed", type=int, default=42, help="random_seed"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="overwrite_cache"
    )    
    parser.add_argument(
        "--k_fold_num", type=int, default=10, help="k_fold_num"
    )
    parser.add_argument(
        "--epoch_num", type=int, default=100, help="epoch_num"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="patience"
    )
    parser.add_argument(
        "--num_works", type=int, default=4, help="num_works"
    )
    parser.add_argument(
        "--num_trials", type=int, default=100, help="num_trials"
    )
    parser.add_argument(
        "--prj_path", type=str, default='', help="prj_path"
    )
    parser.add_argument(
        "--label_path", type=str, default='', help="label_path"
    )
    parser.add_argument(
        "--feature_path", type=str, default='', help="feature_path"
    )
    parser.add_argument(
        "--optuna_log_path", type=str, default='', help="optuna_log_path"
    )
    parser.add_argument(
        "--pretrained_data_path", type=str, default='', help="pretrained_data_path"
    )
    parser.add_argument(
        "--monitor", type=str, default='auroc', help="monitor"
    )
    parser.add_argument(
        "--llama_size", type=str, default='2-7b', help="llama_size"
    )
    parser.add_argument(
        "--name_only", action="store_true", help="Text name_only"
    )
    parser.add_argument(
        "--name_erased", action="store_true", help="Text with out name"
    )
    parser.add_argument(
        "--negtive_sample", type=str, default='all', choices= ['none','equal','all'], help="negtive_sample"
    )
    parser.add_argument(
        "--opt_dir", type=str, default='maximize', choices= ['maximize','minimize'], help="optimize direction"
    )
    args = parser.parse_args()

    freeze_seed(args.random_seed)

    args.prj_path = prj_path
    args.epoch_num = 1 # for xgboost, only 1 epoch is needed
    if args.optuna_log_path.split('_')[-1] == 'NC':
        args.k_fold_num = 10
    print(args)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for k in range(args.k_fold_num):
        # setup fold logger
        fold_logger = set_foldlogger(args, k)
        optuna.logging.get_logger("optuna").handlers = []
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(FoldLoggerStream(fold_logger)))
        study = optuna.create_study(direction=args.opt_dir, sampler = optuna.samplers.RandomSampler()) # maximize/minimize
        fold_logger.info("Start optimization.")

        print(f'----------------current running fold: {k}----------------')
        # init data
        train_tuple, valid_tuple = init_folddata(args, k)
        global best_best_score 
        best_best_score = -np.inf if args.opt_dir == 'maximize' else np.inf # saveflag = False
        # define objective function        
        def objective(trial):
            _model = None
            _train_fprs, _train_tprs, _train_thresholds, _train_pres, _train_recs, _train_thresholds_prc, _train_auc, _train_tn, _train_fp, _train_fn, _train_tp, _train_acc, _train_mcc, _train_precision, _train_recall, _train_specificity, _train_sensitivity, _train_f1, _train_prauc, _train_av_prc, _train_macro_f1, _train_kappa, _train_rmse, _train_mae, _train_pcc = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
            _valid_fprs, _valid_tprs, _valid_thresholds, _valid_pres, _valid_recs, _valid_thresholds_prc, _valid_auc, _valid_tn, _valid_fp, _valid_fn, _valid_tp, _valid_acc, _valid_mcc, _valid_precision, _valid_recall, _valid_specificity, _valid_sensitivity, _valid_f1, _valid_prauc, _valid_av_prc, _valid_macro_f1, _valid_kappa, _valid_rmse, _valid_mae, _valid_pcc = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
            _train_logits, _valid_logits = {}, {}
            best_score = -np.inf if args.opt_dir == 'maximize' else np.inf
            patience = 0

            hpps = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10, step=1),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0, step=0.1),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10, step=1),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
                "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 10, step=1),
                "max_delta_step": trial.suggest_int("max_delta_step", 1, 10, step=1),}
            
            fold_logger.info(f"*-*-*-*-*-*-*-*-*-*")
            
            for epoch in range(args.epoch_num):
                fold_logger.info("Epoch: {}".format(epoch))
                start_time = time.time()

                (model,
                logits_train_tuple, logits_valid_tuple, 
                (fprs_train, tprs_train, thresholds_train, pres_train, recs_train, thresholds_prc_train, 
                    auc_train, tn_train, fp_train, fn_train, tp_train, acc_train, mcc_train, precision_train, recall_train, specificity_train, sensitivity_train, f1_train, prauc_train, av_prc_train, macro_f1_train, kappa_train, rmse_train, mae_train, pcc_train
                    ), 
                (fprs_valid, tprs_valid, thresholds_valid, pres_valid, recs_valid, thresholds_prc_valid, 
                    auc_valid, tn_valid, fp_valid, fn_valid, tp_valid, acc_valid, mcc_valid, precision_valid, recall_valid, specificity_valid, sensitivity_valid, f1_valid, prauc_valid, av_prc_valid, macro_f1_valid, kappa_valid, rmse_valid, mae_valid, pcc_valid
                    ) 
                )  = evaluate_xgb(train_tuple, valid_tuple, args.num_works, hpps)
                
                fold_logger.info(f"train results:  ACC={acc_train:.4f}, MCC={mcc_train:.4f}, Recall={recall_train:.4f}, Precision={precision_train:.4f}, Specificity={specificity_train:.4f}, AUC={auc_train:.4f}, AUPR={prauc_train:.4f}, F1={f1_train:.4f}, macro_F1={macro_f1_train:.4f}, Kappa={kappa_train:.4f}, RMSE={rmse_train:.4f}, MAE={mae_train:.4f}, PCC={pcc_train:.4f}")
                fold_logger.info(f"valid results:  ACC={acc_valid:.4f}, MCC={mcc_valid:.4f}, Recall={recall_valid:.4f}, Precision={precision_valid:.4f}, Specificity={specificity_valid:.4f}, AUC={auc_valid:.4f}, AUPR={prauc_valid:.4f}, F1={f1_valid:.4f}, macro_F1={macro_f1_valid:.4f}, Kappa={kappa_valid:.4f}, RMSE={rmse_valid:.4f}, MAE={mae_valid:.4f}, PCC={pcc_valid:.4f}")
                
                if args.monitor == "auroc":
                    cur_score = auc_valid
                elif args.monitor == "aupr":
                    cur_score = prauc_valid
                elif args.monitor == "f1":
                    cur_score = f1_valid
                elif args.monitor == "mcc":
                    cur_score = mcc_valid
                elif args.monitor == "acc":
                    cur_score = acc_valid
                elif args.monitor == "rmse":
                    cur_score = rmse_valid
                else:
                    raise ValueError("Invalid monitor type!")


                if cur_score > best_score if args.opt_dir == 'maximize' else cur_score < best_score:
                    _model = model
                    _train_fprs, _train_tprs, _train_thresholds, _train_pres, _train_recs, _train_thresholds_prc, _train_auc, _train_tn, _train_fp, _train_fn, _train_tp, _train_acc, _train_mcc, _train_precision, _train_recall, _train_specificity, _train_sensitivity, _train_f1, _train_prauc, _train_av_prc, _train_macro_f1, _train_kappa, _train_rmse, _train_mae, _train_pcc = {'xgboost': fprs_train}, {'xgboost': tprs_train}, {'xgboost': thresholds_train}, {'xgboost': pres_train}, {'xgboost': recs_train}, {'xgboost': thresholds_prc_train}, {'xgboost': auc_train}, {'xgboost': tn_train}, {'xgboost': fp_train}, {'xgboost': fn_train}, {'xgboost': tp_train}, {'xgboost': acc_train}, {'xgboost': mcc_train}, {'xgboost': precision_train}, {'xgboost': recall_train}, {'xgboost': specificity_train}, {'xgboost': sensitivity_train}, {'xgboost': f1_train}, {'xgboost': prauc_train}, {'xgboost': av_prc_train}, {'xgboost': macro_f1_train}, {'xgboost': kappa_train}, {'xgboost': rmse_train}, {'xgboost': mae_train}, {'xgboost': pcc_train}
                    _valid_fprs, _valid_tprs, _valid_thresholds, _valid_pres, _valid_recs, _valid_thresholds_prc, _valid_auc, _valid_tn, _valid_fp, _valid_fn, _valid_tp, _valid_acc, _valid_mcc, _valid_precision, _valid_recall, _valid_specificity, _valid_sensitivity, _valid_f1, _valid_prauc, _valid_av_prc, _valid_macro_f1, _valid_kappa, _valid_rmse, _valid_mae, _valid_pcc = {'xgboost': fprs_valid}, {'xgboost': tprs_valid}, {'xgboost': thresholds_valid}, {'xgboost': pres_valid}, {'xgboost': recs_valid}, {'xgboost': thresholds_prc_valid}, {'xgboost': auc_valid}, {'xgboost': tn_valid}, {'xgboost': fp_valid}, {'xgboost': fn_valid}, {'xgboost': tp_valid}, {'xgboost': acc_valid}, {'xgboost': mcc_valid}, {'xgboost': precision_valid}, {'xgboost': recall_valid}, {'xgboost': specificity_valid}, {'xgboost': sensitivity_valid}, {'xgboost': f1_valid}, {'xgboost': prauc_valid}, {'xgboost': av_prc_valid}, {'xgboost': macro_f1_valid}, {'xgboost': kappa_valid}, {'xgboost': rmse_valid}, {'xgboost': mae_valid}, {'xgboost': pcc_valid}
                    _train_logits, _valid_logits = {'xgboost': logits_train_tuple}, {'xgboost': logits_valid_tuple}
                    best_score = cur_score
                    patience = 0
                else:
                    patience += 1
                    if patience > args.patience:
                        # print("Early Stopping")
                        fold_logger.info("Early Stopping")
                        break
            
                fold_logger.info(f"Epoch: {epoch} finished, time used: {time.time()-start_time:.4f}s")
            
            fold_logger.info(f"Current trial hyper-parameters: {model.get_params()}")

            result = (
                    (_model,),
                    (_train_fprs, _train_tprs, _train_thresholds, _train_pres, _train_recs, _train_thresholds_prc, _train_auc, _train_tn, _train_fp, _train_fn, _train_tp, _train_acc, _train_mcc, _train_precision, _train_recall, _train_specificity, _train_sensitivity, _train_f1, _train_prauc, _train_av_prc, _train_macro_f1, _train_kappa, _train_rmse, _train_mae, _train_pcc), 
                    (_valid_fprs, _valid_tprs, _valid_thresholds, _valid_pres, _valid_recs, _valid_thresholds_prc, _valid_auc, _valid_tn, _valid_fp, _valid_fn, _valid_tp, _valid_acc, _valid_mcc, _valid_precision, _valid_recall, _valid_specificity, _valid_sensitivity, _valid_f1, _valid_prauc, _valid_av_prc, _valid_macro_f1, _valid_kappa, _valid_rmse, _valid_mae, _valid_pcc), 
                    (_train_logits, _valid_logits)
                    )
            # save results
            global best_best_score
            if best_score > best_best_score if args.opt_dir == 'maximize' else best_score < best_best_score:
                best_best_score = best_score
                print(f"*-*-*-*-*-*-*-*-*-*")
                print(f"all epochs finished, Epoch Best Score updated and save to cache......: {args.monitor}: {best_score}")
                print(f"train results:  ACC={_train_acc['xgboost']:.4f}, MCC={_train_mcc['xgboost']:.4f}, Recall={_train_recall['xgboost']:.4f}, Precision={_train_precision['xgboost']:.4f}, Specificity={_train_specificity['xgboost']:.4f}, AUC={_train_auc['xgboost']:.4f}, AUPR={_train_prauc['xgboost']:.4f}, F1={_train_f1['xgboost']:.4f}, macro_F1={_train_macro_f1['xgboost']:.4f}, Kappa={_train_kappa['xgboost']:.4f}, RMSE={_train_rmse['xgboost']:.4f}, MAE={_train_mae['xgboost']:.4f}, PCC={_train_pcc['xgboost']:.4f}")
                print(f"valid results:  ACC={_valid_acc['xgboost']:.4f}, MCC={_valid_mcc['xgboost']:.4f}, Recall={_valid_recall['xgboost']:.4f}, Precision={_valid_precision['xgboost']:.4f}, Specificity={_valid_specificity['xgboost']:.4f}, AUC={_valid_auc['xgboost']:.4f}, AUPR={_valid_prauc['xgboost']:.4f}, F1={_valid_f1['xgboost']:.4f}, macro_F1={_valid_macro_f1['xgboost']:.4f}, Kappa={_valid_kappa['xgboost']:.4f}, RMSE={_valid_rmse['xgboost']:.4f}, MAE={_valid_mae['xgboost']:.4f}, PCC={_valid_pcc['xgboost']:.4f}")
                superIO(args, result, k, hpps)

            # optuna monitor
            return best_score

        # model running
        study.optimize(objective, n_trials=args.num_trials)

        fold_logger.info("*****OPTUNA FINISHED FOR FOLD {}*****".format(k))
        fold_logger.info("Finished optimization with best score: {}".format(study.best_value))

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("{}: {}".format(key, value))

        print(f'----------------finish running fold: {k}----------------')

        
    
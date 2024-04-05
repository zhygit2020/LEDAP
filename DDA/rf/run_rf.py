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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
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
    # return np.load(path / name / f'{name}.npy')[...,0,:]
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
    # load data
    with open(Path(str(args.label_path)+f'/fold{k}_hetionet_graph_data+treats') / f'train-valid-test_{k}.pkl', 'rb') as f:
        data = pickle.load(f)
    train_data, valid_data, test_data = data['train'], data['valid'], data['test']
    print('data loaded')
    if args.negtive_sample == 'equal':
        pass # train_data = train_data[train_data['label'] == 0].sample(n=len(train_data[train_data['label'] != 0])).append(train_data[train_data['label'] != 0])
    elif args.negtive_sample == 'none':
        pass # train_data = train_data[train_data['label'] != 0]
    elif args.negtive_sample == 'all':
        pass
    # load embedding
    if args.name_only:
        compoundfea_path = Path(str(args.feature_path)+f'/llama_{args.llama_size}') / str('Compound_'+args.optuna_log_path.split('_')[-1].lower()+'_nameonly')
        diseasefea_path = Path(str(args.feature_path)+f'/llama_{args.llama_size}') / str('Disease_'+args.optuna_log_path.split('_')[-1].lower()+'_nameonly')
    elif args.name_erased:
        compoundfea_path = Path(str(args.feature_path)+f'/llama_{args.llama_size}') / str('Compound_'+args.optuna_log_path.split('_')[-1].lower()+'_nameerased')
        diseasefea_path = Path(str(args.feature_path)+f'/llama_{args.llama_size}') / str('Disease_'+args.optuna_log_path.split('_')[-1].lower()+'_nameerased')
    else: 
        compoundfea_path = Path(str(args.feature_path)+f'/llama_{args.llama_size}') / str('Compound_'+args.optuna_log_path.split('_')[-1].lower())
        diseasefea_path = Path(str(args.feature_path)+f'/llama_{args.llama_size}') / str('Disease_'+args.optuna_log_path.split('_')[-1].lower())

    # train
    pool = multiprocessing.Pool(processes=args.num_works)
    train_src = pool.map(partial(load_and_process_seqfea,compoundfea_path),train_data['src_kind']+"::"+train_data['src_identifier'])
    train_tar = pool.map(partial(load_and_process_seqfea,diseasefea_path),train_data['tar_kind']+"::"+train_data['tar_identifier'])
    pool.close()
    pool.join()
    train = np.concatenate([np.array(train_src,dtype=np.float32), np.array(train_tar,dtype=np.float32)], axis=1)
    train_label = np.array(train_data['Label'], dtype=np.int64)
    # valid
    pool = multiprocessing.Pool(processes=args.num_works)
    valid_src = pool.map(partial(load_and_process_seqfea,compoundfea_path),valid_data['src_kind']+"::"+valid_data['src_identifier'])
    valid_tar = pool.map(partial(load_and_process_seqfea,diseasefea_path),valid_data['tar_kind']+"::"+valid_data['tar_identifier'])
    pool.close()
    pool.join()
    valid = np.concatenate([np.array(valid_src,dtype=np.float32), np.array(valid_tar,dtype=np.float32)], axis=1)
    valid_label = np.array(valid_data['Label'], dtype=np.int64)
    # test
    pool = multiprocessing.Pool(processes=args.num_works)
    test_src = pool.map(partial(load_and_process_seqfea,compoundfea_path),test_data['src_kind']+"::"+test_data['src_identifier'])
    test_tar = pool.map(partial(load_and_process_seqfea,diseasefea_path),test_data['tar_kind']+"::"+test_data['tar_identifier'])
    pool.close()
    pool.join()
    test = np.concatenate([np.array(test_src,dtype=np.float32), np.array(test_tar,dtype=np.float32)], axis=1)
    test_label = np.array(test_data['Label'], dtype=np.int64)

    # random.shuffle(zip(train, train_label))
    index = np.arange(train_label.shape[0])
    np.random.shuffle(index)
    train, train_label = train[index].squeeze(), train_label[index].squeeze()

    mean, std = standard_scale((train, valid, test))
    train = (train - mean) / std
    valid = (valid - mean) / std
    test = (test - mean) / std

    print("mean.shape, std.shape: ", mean.shape, std.shape)
    print("mean & std saved to: ", str(Path(str(args.label_path)+f'/fold{k}_hetionet_graph_data+treats')/'mean.npy'), ' & ' , str(Path(str(args.label_path)+f'/fold{k}_hetionet_graph_data+treats')/'std.npy'))
    np.save(Path(str(args.label_path)+f'/fold{k}_hetionet_graph_data+treats') / 'mean.npy', mean)
    np.save(Path(str(args.label_path)+f'/fold{k}_hetionet_graph_data+treats') / 'std.npy', std)
    print('data inited')
    print(f'inited data info:')
    print(f'train: {train.shape}, valid: {valid.shape}, test: {test.shape}')
    print(f"memory used: train={train.nbytes/(1024**3)} GB, valid={valid.nbytes/(1024**3)} GB, test={test.nbytes/(1024**3)} GB")
    return (train, train_label), (valid, valid_label), (test, test_label)

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
            pd.DataFrame({'auc':results[3][6][etk], 'tn':results[3][7][etk], 'fp':results[3][8][etk], 'fn':results[3][9][etk], 'tp':results[3][10][etk], 'acc':results[3][11][etk], 'mcc':results[3][12][etk], 'precision':results[3][13][etk], 'recall':results[3][14][etk], 'specificity':results[3][15][etk], 'sensitivity':results[3][16][etk], 'f1':results[3][17][etk], 'prauc':results[3][18][etk], 'av_prc':results[3][19][etk], 'macro_f1':results[3][20][etk], 'kappa':results[3][21][etk], 'rmse':results[3][22][etk], 'mae':results[3][23][etk], 'pcc':results[3][24][etk]}, index=[etk]).to_csv(metric_savepath / f'test_metrics_of_{etk}_for_{fold}th_Fold.csv')

    def save_logits(args, fold, results):
        logit_savepath = saveio_path / 'logit_data' / f'{fold}th_Fold'
        logit_savepath.mkdir(parents=True, exist_ok=True)
        for etk in results[4][0]:
            pd.DataFrame({'y_true_train_logits':results[4][0][etk][0], 'y_scores_train_logits':[str(i) for i in results[4][0][etk][1]], 'threshold_train_logits':np.array([results[4][0][etk][2] for i in range(len(results[4][0][etk][0]))])}).to_csv(logit_savepath / f'train_logits_of_{etk}_for_{fold}th_Fold.csv')
            pd.DataFrame({'y_true_valid_logits':results[4][1][etk][0], 'y_scores_valid_logits':[str(i) for i in results[4][1][etk][1]], 'threshold_valid_logits':np.array([results[4][1][etk][2] for i in range(len(results[4][1][etk][0]))])}).to_csv(logit_savepath / f'valid_logits_of_{etk}_for_{fold}th_Fold.csv')
            pd.DataFrame({'y_true_test_logits':results[4][2][etk][0], 'y_scores_test_logits':[str(i) for i in results[4][2][etk][1]], 'threshold_test_logits':np.array([results[4][2][etk][2] for i in range(len(results[4][2][etk][0]))])}).to_csv(logit_savepath / f'test_logits_of_{etk}_for_{fold}th_Fold.csv')

    # save best kth model
    save_model(args, fold, results[0][0], optuna_hpps)
    # save best kth Fold ROC data
    save_curves(args, fold, results[1][0], results[1][1], results[1][2], flag='train_ROC')
    save_curves(args, fold, results[2][0], results[2][1], results[2][2], flag='valid_ROC')
    save_curves(args, fold, results[3][0], results[3][1], results[3][2], flag='test_ROC')

    # save best kth Fold PRC data
    save_curves(args, fold, results[1][3], results[1][4], results[1][5], flag='train_PRC')
    save_curves(args, fold, results[2][3], results[2][4], results[2][5], flag='valid_PRC')
    save_curves(args, fold, results[3][3], results[3][4], results[3][5], flag='test_PRC')

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

def evaluate_rf(train, valid, test, num_workers, optuna_hpps):
    # create model instance
    bst = RandomForestClassifier(n_jobs=num_workers, random_state=args.random_seed, **optuna_hpps)

    # fit model
    bst.fit(train[0], train[1])
    # make predictions
    train_preds = bst.predict_proba(train[0])
    valid_preds = bst.predict_proba(valid[0])
    test_preds = bst.predict_proba(test[0])

    threshold_tr = 0.0
    threshold_v = 0.0
    threshold_t = 0.0

    return bst, (train[1], train_preds, threshold_tr), (valid[1], valid_preds, threshold_v), (test[1], test_preds, threshold_t), calculate_fusion_scores(train[1], train_preds, threshold_tr), calculate_fusion_scores(valid[1], valid_preds, threshold_v), calculate_fusion_scores(test[1], test_preds, threshold_t)



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
    args.epoch_num = 1 # for randomforrest, only 1 epoch is needed
    print(args)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    import warnings
    # ignore RuntimeWarning
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for k in range(args.k_fold_num):
        # setup fold logger
        fold_logger = set_foldlogger(args, k)
        optuna.logging.get_logger("optuna").handlers = []
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(FoldLoggerStream(fold_logger)))
        study = optuna.create_study(direction=args.opt_dir, sampler = optuna.samplers.RandomSampler())
        fold_logger.info("Start optimization.")

        print(f'----------------current running fold: {k}----------------')
        # init data
        train_tuple, valid_tuple, test_tuple = init_folddata(args, k)
        global best_best_score 
        best_best_score = -np.inf if args.opt_dir == 'maximize' else np.inf # saveflag = False
        # define objective function        
        def objective(trial):
            _model = None
            _train_fprs, _train_tprs, _train_thresholds, _train_pres, _train_recs, _train_thresholds_prc, _train_auc, _train_tn, _train_fp, _train_fn, _train_tp, _train_acc, _train_mcc, _train_precision, _train_recall, _train_specificity, _train_sensitivity, _train_f1, _train_prauc, _train_av_prc, _train_macro_f1, _train_kappa, _train_rmse, _train_mae, _train_pcc = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
            _valid_fprs, _valid_tprs, _valid_thresholds, _valid_pres, _valid_recs, _valid_thresholds_prc, _valid_auc, _valid_tn, _valid_fp, _valid_fn, _valid_tp, _valid_acc, _valid_mcc, _valid_precision, _valid_recall, _valid_specificity, _valid_sensitivity, _valid_f1, _valid_prauc, _valid_av_prc, _valid_macro_f1, _valid_kappa, _valid_rmse, _valid_mae, _valid_pcc = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
            _test_fprs, _test_tprs, _test_thresholds, _test_pres, _test_recs, _test_thresholds_prc, _test_auc, _test_tn, _test_fp, _test_fn, _test_tp, _test_acc, _test_mcc, _test_precision, _test_recall, _test_specificity, _test_sensitivity, _test_f1, _test_prauc, _test_av_prc, _test_macro_f1, _test_kappa, _test_rmse, _test_mae, _test_pcc = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
            _train_logits, _valid_logits, _test_logits = {}, {}, {}
            best_score = -np.inf if args.opt_dir == 'maximize' else np.inf
            patience = 0

            hpps = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "max_depth": trial.suggest_int("max_depth", 3, 10, step=1),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, step=1),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10, step=1),
                "min_weight_fraction_leaf": trial.suggest_loguniform("min_weight_fraction_leaf", 1e-8, 1e-2),
                "max_features": trial.suggest_categorical("max_features", ['auto', 'sqrt', 'log2']),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 100, step=10),
                "min_impurity_decrease": trial.suggest_loguniform("min_impurity_decrease", 1e-8, 1e-2),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
                "ccp_alpha": trial.suggest_loguniform("ccp_alpha", 1e-8, 1e-2),
                "max_samples": trial.suggest_int("max_samples", 10, 100, step=10),
            }
            if trial.suggest_categorical("bootstrap", [True, False]):
                hpps["bootstrap"]=True
                hpps["oob_score"]=True
            else:
                hpps["bootstrap"]=False
                hpps["oob_score"]=False

            fold_logger.info(f"*-*-*-*-*-*-*-*-*-*")
            
            for epoch in range(args.epoch_num):
                fold_logger.info("Epoch: {}".format(epoch))
                start_time = time.time()

                (model,
                logits_train_tuple, logits_valid_tuple, logits_test_tuple, 
                (fprs_train, tprs_train, thresholds_train, pres_train, recs_train, thresholds_prc_train, 
                    auc_train, tn_train, fp_train, fn_train, tp_train, acc_train, mcc_train, precision_train, recall_train, specificity_train, sensitivity_train, f1_train, prauc_train, av_prc_train, macro_f1_train, kappa_train, rmse_train, mae_train, pcc_train
                    ), 
                (fprs_valid, tprs_valid, thresholds_valid, pres_valid, recs_valid, thresholds_prc_valid, 
                    auc_valid, tn_valid, fp_valid, fn_valid, tp_valid, acc_valid, mcc_valid, precision_valid, recall_valid, specificity_valid, sensitivity_valid, f1_valid, prauc_valid, av_prc_valid, macro_f1_valid, kappa_valid, rmse_valid, mae_valid, pcc_valid
                    ), 
                (fprs_test, tprs_test, thresholds_test, pres_test, recs_test, thresholds_prc_test, 
                    auc_test, tn_test, fp_test, fn_test, tp_test, acc_test, mcc_test, precision_test, recall_test, specificity_test, sensitivity_test, f1_test, prauc_test, av_prc_test, macro_f1_test, kappa_test, rmse_test, mae_test, pcc_test
                    ) 
                )  = evaluate_rf(train_tuple, valid_tuple, test_tuple, args.num_works, hpps)

                fold_logger.info(f"train results:  ACC={acc_train:.4f}, MCC={mcc_train:.4f}, Recall={recall_train:.4f}, Precision={precision_train:.4f}, Specificity={specificity_train:.4f}, AUC={auc_train:.4f}, AUPR={prauc_train:.4f}, F1={f1_train:.4f}, macro_F1={macro_f1_train:.4f}, Kappa={kappa_train:.4f}, RMSE={rmse_train:.4f}, MAE={mae_train:.4f}, PCC={pcc_train:.4f}")
                fold_logger.info(f"valid results:  ACC={acc_valid:.4f}, MCC={mcc_valid:.4f}, Recall={recall_valid:.4f}, Precision={precision_valid:.4f}, Specificity={specificity_valid:.4f}, AUC={auc_valid:.4f}, AUPR={prauc_valid:.4f}, F1={f1_valid:.4f}, macro_F1={macro_f1_valid:.4f}, Kappa={kappa_valid:.4f}, RMSE={rmse_valid:.4f}, MAE={mae_valid:.4f}, PCC={pcc_valid:.4f}")
                fold_logger.info(f"test results:  ACC={acc_test:.4f}, MCC={mcc_test:.4f}, Recall={recall_test:.4f}, Precision={precision_test:.4f}, Specificity={specificity_test:.4f}, AUC={auc_test:.4f}, AUPR={prauc_test:.4f}, F1={f1_test:.4f}, macro_F1={macro_f1_test:.4f}, Kappa={kappa_test:.4f}, RMSE={rmse_test:.4f}, MAE={mae_test:.4f}, PCC={pcc_test:.4f}")
                
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
                    _train_fprs, _train_tprs, _train_thresholds, _train_pres, _train_recs, _train_thresholds_prc, _train_auc, _train_tn, _train_fp, _train_fn, _train_tp, _train_acc, _train_mcc, _train_precision, _train_recall, _train_specificity, _train_sensitivity, _train_f1, _train_prauc, _train_av_prc, _train_macro_f1, _train_kappa, _train_rmse, _train_mae, _train_pcc = {'randomforrest': fprs_train}, {'randomforrest': tprs_train}, {'randomforrest': thresholds_train}, {'randomforrest': pres_train}, {'randomforrest': recs_train}, {'randomforrest': thresholds_prc_train}, {'randomforrest': auc_train}, {'randomforrest': tn_train}, {'randomforrest': fp_train}, {'randomforrest': fn_train}, {'randomforrest': tp_train}, {'randomforrest': acc_train}, {'randomforrest': mcc_train}, {'randomforrest': precision_train}, {'randomforrest': recall_train}, {'randomforrest': specificity_train}, {'randomforrest': sensitivity_train}, {'randomforrest': f1_train}, {'randomforrest': prauc_train}, {'randomforrest': av_prc_train}, {'randomforrest': macro_f1_train}, {'randomforrest': kappa_train}, {'randomforrest': rmse_train}, {'randomforrest': mae_train}, {'randomforrest': pcc_train}
                    _valid_fprs, _valid_tprs, _valid_thresholds, _valid_pres, _valid_recs, _valid_thresholds_prc, _valid_auc, _valid_tn, _valid_fp, _valid_fn, _valid_tp, _valid_acc, _valid_mcc, _valid_precision, _valid_recall, _valid_specificity, _valid_sensitivity, _valid_f1, _valid_prauc, _valid_av_prc, _valid_macro_f1, _valid_kappa, _valid_rmse, _valid_mae, _valid_pcc = {'randomforrest': fprs_valid}, {'randomforrest': tprs_valid}, {'randomforrest': thresholds_valid}, {'randomforrest': pres_valid}, {'randomforrest': recs_valid}, {'randomforrest': thresholds_prc_valid}, {'randomforrest': auc_valid}, {'randomforrest': tn_valid}, {'randomforrest': fp_valid}, {'randomforrest': fn_valid}, {'randomforrest': tp_valid}, {'randomforrest': acc_valid}, {'randomforrest': mcc_valid}, {'randomforrest': precision_valid}, {'randomforrest': recall_valid}, {'randomforrest': specificity_valid}, {'randomforrest': sensitivity_valid}, {'randomforrest': f1_valid}, {'randomforrest': prauc_valid}, {'randomforrest': av_prc_valid}, {'randomforrest': macro_f1_valid}, {'randomforrest': kappa_valid}, {'randomforrest': rmse_valid}, {'randomforrest': mae_valid}, {'randomforrest': pcc_valid}
                    _test_fprs, _test_tprs, _test_thresholds, _test_pres, _test_recs, _test_thresholds_prc, _test_auc, _test_tn, _test_fp, _test_fn, _test_tp, _test_acc, _test_mcc, _test_precision, _test_recall, _test_specificity, _test_sensitivity, _test_f1, _test_prauc, _test_av_prc, _test_macro_f1, _test_kappa, _test_rmse, _test_mae, _test_pcc = {'randomforrest': fprs_test}, {'randomforrest': tprs_test}, {'randomforrest': thresholds_test}, {'randomforrest': pres_test}, {'randomforrest': recs_test}, {'randomforrest': thresholds_prc_test}, {'randomforrest': auc_test}, {'randomforrest': tn_test}, {'randomforrest': fp_test}, {'randomforrest': fn_test}, {'randomforrest': tp_test}, {'randomforrest': acc_test}, {'randomforrest': mcc_test}, {'randomforrest': precision_test}, {'randomforrest': recall_test}, {'randomforrest': specificity_test}, {'randomforrest': sensitivity_test}, {'randomforrest': f1_test}, {'randomforrest': prauc_test}, {'randomforrest': av_prc_test}, {'randomforrest': macro_f1_test}, {'randomforrest': kappa_test}, {'randomforrest': rmse_test}, {'randomforrest': mae_test}, {'randomforrest': pcc_test}
                    _train_logits, _valid_logits, _test_logits = {'randomforrest': logits_train_tuple}, {'randomforrest': logits_valid_tuple}, {'randomforrest': logits_test_tuple}
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
                    (_test_fprs, _test_tprs, _test_thresholds, _test_pres, _test_recs, _test_thresholds_prc, _test_auc, _test_tn, _test_fp, _test_fn, _test_tp, _test_acc, _test_mcc, _test_precision, _test_recall, _test_specificity, _test_sensitivity, _test_f1, _test_prauc, _test_av_prc, _test_macro_f1, _test_kappa, _test_rmse, _test_mae, _test_pcc), 
                    (_train_logits, _valid_logits, _test_logits)
                    )
            # save results
            global best_best_score
            if best_score > best_best_score if args.opt_dir == 'maximize' else best_score < best_best_score: # if save_flag
                best_best_score = best_score
                print(f"*-*-*-*-*-*-*-*-*-*")
                print(f"all epochs finished, Epoch Best Score updated and save to cache......: {args.monitor}: {best_score}")
                print(f"train results:  ACC={_train_acc['randomforrest']:.4f}, MCC={_train_mcc['randomforrest']:.4f}, Recall={_train_recall['randomforrest']:.4f}, Precision={_train_precision['randomforrest']:.4f}, Specificity={_train_specificity['randomforrest']:.4f}, AUC={_train_auc['randomforrest']:.4f}, AUPR={_train_prauc['randomforrest']:.4f}, F1={_train_f1['randomforrest']:.4f}, macro_F1={_train_macro_f1['randomforrest']:.4f}, Kappa={_train_kappa['randomforrest']:.4f}, RMSE={_train_rmse['randomforrest']:.4f}, MAE={_train_mae['randomforrest']:.4f}, PCC={_train_pcc['randomforrest']:.4f}")
                print(f"valid results:  ACC={_valid_acc['randomforrest']:.4f}, MCC={_valid_mcc['randomforrest']:.4f}, Recall={_valid_recall['randomforrest']:.4f}, Precision={_valid_precision['randomforrest']:.4f}, Specificity={_valid_specificity['randomforrest']:.4f}, AUC={_valid_auc['randomforrest']:.4f}, AUPR={_valid_prauc['randomforrest']:.4f}, F1={_valid_f1['randomforrest']:.4f}, macro_F1={_valid_macro_f1['randomforrest']:.4f}, Kappa={_valid_kappa['randomforrest']:.4f}, RMSE={_valid_rmse['randomforrest']:.4f}, MAE={_valid_mae['randomforrest']:.4f}, PCC={_valid_pcc['randomforrest']:.4f}")
                print(f"test results:  ACC={_test_acc['randomforrest']:.4f}, MCC={_test_mcc['randomforrest']:.4f}, Recall={_test_recall['randomforrest']:.4f}, Precision={_test_precision['randomforrest']:.4f}, Specificity={_test_specificity['randomforrest']:.4f}, AUC={_test_auc['randomforrest']:.4f}, AUPR={_test_prauc['randomforrest']:.4f}, F1={_test_f1['randomforrest']:.4f}, macro_F1={_test_macro_f1['randomforrest']:.4f}, Kappa={_test_kappa['randomforrest']:.4f}, RMSE={_test_rmse['randomforrest']:.4f}, MAE={_test_mae['randomforrest']:.4f}, PCC={_test_pcc['randomforrest']:.4f}")
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

        
    
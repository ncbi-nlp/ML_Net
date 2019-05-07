"""
evaluation metrics, with refrence to scripts from:
https://physionet.org/works/ICD9CodingofDischargeSummaries
"""

import numpy as np
import data_utils as data_utils

ancestors = data_utils.load_obj("tf_data/ancestors.pk")
root = 5367

def only_leaves(codes):
    leaves = []
    # codes = np.array(codes)
    all_ancestors = set()
    for code in codes:
        all_ancestors.update(list(ancestors[code][:-1]))
    for code in codes:
        if code not in all_ancestors:
            leaves.append(code)
    return leaves

def get_p_r_f_jamia(logits, counts, labels):
    ranks = np.argsort(-logits, axis=1)
    FNs = []
    FPs = []
    TPs = []

    predict_all_list = list()
    gold_all_list = list()

    for rank, L, k in zip(ranks, labels, counts):

        predict_id_list = list(rank[:k + 1])
        predict_id_list.append(root)
        predict_id_set = set(predict_id_list)

        # Prune the predictions to respect the conditional classification
        # constraint (all ancestors must be predicted true for a child to be
        # predicted true)

        filtered_predictions = []
        for prediction in predict_id_set:
            if prediction == root or np.all([anc in predict_id_set for anc in ancestors[prediction]]):
                filtered_predictions.append(prediction)
        predict_id_set = set(filtered_predictions)

        full_predict_id_set = set()

        for predict in predict_id_set:
            full_predict_id_set.update(list(ancestors[predict]))

        pred_only_leave = set(only_leaves(full_predict_id_set))

        full_gold_set = set()
        for gs in L:
            full_gold_set.update(list(ancestors[gs]))

        gold_set = set(only_leaves(full_gold_set))

        TP = 0
        FP = 0
        FN = 0
        for code in gold_set:
            if len(set(list(ancestors[code])) - set(full_predict_id_set)) > 0:
                FN += 1
            ## else:
            ##     TP += 1
        for code in pred_only_leave:
            anc_set = set(list(ancestors[code]))
            if len(anc_set - set(full_gold_set)) > 0 and not np.any([x in anc_set for x in gold_set]):
                FP += 1
            else:
                TP += 1
        FNs.append(FN)
        FPs.append(FP)
        TPs.append(TP)

        predict_all_list.append(set(predict_id_set))
        gold_all_list.append(set(L))


    FNs = np.array(FNs, np.float)
    FPs = np.array(FPs, np.float)
    TPs = np.array(TPs, np.float)

    mean_prc = np.nanmean(np.where(TPs + FPs > 0, TPs / (TPs + FPs), 0))
    mean_rec = np.nanmean(np.where(TPs + FNs > 0, TPs / (TPs + FNs), 0))
    f_score = 2 * (mean_prc * mean_rec) / (mean_prc + mean_rec)

    return mean_prc, mean_rec, f_score, predict_all_list, gold_all_list
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
import itertools
from tqdm import tqdm
import time
import cupy as cp 

def sum2score(out_class_sum_list, out_class_cnt_list, inner_class_sum_list, inner_class_cnt_list):
    outer_class_mean_dist = out_class_sum_list / out_class_cnt_list
    inner_class_mean_dist = inner_class_sum_list / (inner_class_cnt_list - 1)
    nan_bool = cp.isnan(inner_class_mean_dist) # The line is time-costing
    idxs = cp.nonzero(nan_bool)
    avg = cp.nanmax(inner_class_mean_dist, axis=1) # Use the max to replace
    inner_class_mean_dist[idxs[0], idxs[1]] = avg[idxs[0]]
    score = cp.mean(outer_class_mean_dist - inner_class_mean_dist, axis=1)
    return score

def calc_new_sum_cnt(new, new2, old = None):
    # new: (beamsize*eqsize)x(n)
    # new2: (beamsize*eqsize)x(deltan)
    # old: (beamsize)x(n-deltan)
    if old is None:
        return new
    if old is not None:
        n = new.shape[1]
        deltan = new2.shape[1]
        eqsize = new.shape[0]//old.shape[0]
        assert(old.shape[1]==n-deltan and new.shape[0]%old.shape[0]==0)
        #final = old.repeat(eqsize, axis=0) # (beamsize*eqsize)x(n-deltan)
        #final += new[:,0:n-deltan] # (beamsize*eqsize)x(n-deltan)
        final = new.reshape((old.shape[0], eqsize, new.shape[1]))[:,:,0:old.shape[1]] + old.reshape((old.shape[0], 1, old.shape[1]))
        final = final.reshape((-1, final.shape[2]))
        final = cp.concatenate((final,new2), axis=1) # (beamsize*eqsize)x(n)
    return final

def score_label_similarity(label_lists_org, pair_dist_org, inner_sum_old = None, inner_cnt_old = None, outer_sum_old = None, outer_cnt_old = None):
    # label_lists: (beamsize*eqsize)x(n)
    # label_lists2: (beamsize*eqsize)x(deltan)
    # pair_dist: (n)x(deltan)
    # out_class_sum_list: (beamsize*eqsize)x(n)
    n = label_lists_org.shape[1]
    if inner_sum_old is None:
        delta_n = n
    else: # Calculate incrementally
        delta_n = n - inner_sum_old.shape[1]
    # Slice pair_distance
    assert(n <= pair_dist_org.shape[0])
    pair_distance = pair_dist_org[0:n, n-delta_n:n]

    label_lists = cp.array(label_lists_org)
    label_lists2 = label_lists[:, -delta_n:]
    pair_dist = cp.array(pair_distance)
    same_matrix = cp.equal(label_lists.reshape(-1, n, 1), label_lists2.reshape(-1, 1, delta_n))
    #diff_matrix = ~same_matrix
    # shape: (beamsize*eqsize)x(n)
    inner_class_sum_list = cp.sum(cp.multiply(same_matrix, pair_dist), axis=2)
    inner_class_cnt_list = cp.sum(same_matrix, axis=2)
    #out_class_sum_list = cp.sum(cp.multiply(diff_matrix, pair_dist), axis=2)
    out_class_sum_list = cp.sum(pair_dist, axis=1) - inner_class_sum_list
    #out_class_cnt_list = cp.sum(diff_matrix, axis=2)
    out_class_cnt_list = same_matrix.shape[2]-inner_class_cnt_list
    # shape: (beamsize*eqsize)x(deltan)
    inner_class_sum_list2 = cp.sum(cp.multiply(same_matrix, pair_dist), axis=1)
    inner_class_cnt_list2 = cp.sum(same_matrix, axis=1)
    #out_class_sum_list2 = cp.sum(cp.multiply(diff_matrix, pair_dist), axis=1)
    out_class_sum_list2 = cp.sum(pair_dist, axis=0) - inner_class_sum_list2
    #out_class_cnt_list2 = cp.sum(diff_matrix, axis=1)
    out_class_cnt_list2 = same_matrix.shape[1]-inner_class_cnt_list2

    out_class_sum_list = calc_new_sum_cnt(out_class_sum_list, out_class_sum_list2, outer_sum_old)
    out_class_cnt_list = calc_new_sum_cnt(out_class_cnt_list, out_class_cnt_list2, outer_cnt_old)
    inner_class_sum_list = calc_new_sum_cnt(inner_class_sum_list, inner_class_sum_list2, inner_sum_old)
    inner_class_cnt_list = calc_new_sum_cnt(inner_class_cnt_list, inner_class_cnt_list2, inner_cnt_old)
    score = sum2score(out_class_sum_list, out_class_cnt_list, inner_class_sum_list, inner_class_cnt_list)
    return score, out_class_sum_list, out_class_cnt_list, inner_class_sum_list, inner_class_cnt_list

def score_label_prob(label_lists, prob_val):
    # Slice prob_val
    assert(label_lists.shape[1] <= len(prob_val))
    prob_val = prob_val[0:label_lists.shape[1]]
    label_lists, prob_val = cp.array(label_lists), cp.array(prob_val)
    probs_list = prob_val[np.arange(len(prob_val)), label_lists]
    log_probs_list = cp.log(probs_list)
    log_prods_list = cp.sum(log_probs_list, axis=1, dtype = cp.float64)
    prods_list = cp.exp(log_prods_list, dtype = cp.float64)
    score_list = prods_list #/np.sum(prods_list)
    return score_list

def list_split(list1, abduced_list):
    ret = []
    cur = 0
    for i in range(len(abduced_list)):
        ret.append(list1[cur:cur+len(abduced_list[i][0])])
        cur += len(abduced_list[i][0])
    return ret

def select_abduced_result(pair_distance, prob_val, abduced_result, labeled_y, ground_label = None, beam_width = None, similar_coef = 1, inner_sum_old = None, inner_cnt_old = None, outer_sum_old = None, outer_cnt_old = None):
    '''
    inner_sum_old: (beamsize)x(n-deltan)
    inner_cnt_old: (beamsize)x(n-deltan)
    outer_sum_old: (beamsize)x(n-deltan)
    outer_cnt_old: (beamsize)x(n-deltan)
    abduced_result:(beamsize*eqsize)x(n)
    '''
    out_class_sum, out_class_cnt, inner_class_sum, inner_class_cnt = inner_sum_old, inner_cnt_old, outer_sum_old, outer_cnt_old
    # Only one abduced result
    if len(abduced_result) == 1:
        return abduced_result, abduced_result[0], inner_sum_old, inner_cnt_old, outer_sum_old, outer_cnt_old
    # Slice prob_val
    assert(abduced_result.shape[1] <= len(prob_val))
    prob_val = prob_val[0:abduced_result.shape[1]]

    # Score each abduced result and select the best
    if similar_coef > 0:
        score_similarity_org_list, out_class_sum, out_class_cnt, inner_class_sum, inner_class_cnt = score_label_similarity(abduced_result, pair_distance, inner_sum_old, inner_cnt_old, outer_sum_old, outer_cnt_old)
        score_similarity_list = (score_similarity_org_list-score_similarity_org_list.mean())/score_similarity_org_list.std() # scale
    if similar_coef < 1:
        score_prob_org_list = score_label_prob(abduced_result, prob_val)
        score_prob_list = (score_prob_org_list-score_prob_org_list.mean())/score_prob_org_list.std() # scale
    if similar_coef == 0:
        score_list = score_prob_list
    elif similar_coef == 1:
        score_list = score_similarity_list
    else:
        score_list = similar_coef * score_similarity_list + (1 - similar_coef) * score_prob_list
    score_list = score_list.get()  # TO CPU
    best = np.argmax(score_list)
    #print('best   score', similar_coef*score_similarity_org_list[best]+(1-similar_coef)*score_prob_org_list[best], score_similarity_org_list[best], score_prob_org_list[best], list(abduced_result[best][len(labeled_y):]))
    #ground_all = np.array([labeled_y + ground_label])
    #print('ground score', similar_coef*score_label_similarity(ground_all, pair_distance)[0][0]+(1-similar_coef)*score_label_prob(ground_all, prob_val)[0],score_label_similarity(ground_all, pair_distance)[0][0], score_label_prob(ground_all, prob_val)[0], ground_label)
    #input()
    if beam_width == None:
        return None, abduced_result[best], None, None, None, None
    # Beam search
    if len(score_list) <= beam_width:
        return abduced_result, abduced_result[best], out_class_sum, out_class_cnt, inner_class_sum, inner_class_cnt
    top_k_score_idxs = np.argpartition(-np.array(score_list), beam_width)[0:beam_width]
    if similar_coef > 0:
        return abduced_result[top_k_score_idxs], abduced_result[best], out_class_sum[top_k_score_idxs], out_class_cnt[top_k_score_idxs], inner_class_sum[top_k_score_idxs], inner_class_cnt[top_k_score_idxs]
    else:
        return abduced_result[top_k_score_idxs], abduced_result[best], None, None, None, None

def nn_select_batch_abduced_result(model, labeled_X, labeled_y, predict_probs_list, predict_fea_list, abduced_list, abduction_batch_size = 3, ground_labels_list = None, beam_width= None, similar_coef = 0.9):
    print("Getting labeled data's prob and feature")
    if labeled_X is not None:
        prob_val_labeled_list, dense_val_labeled_list = model.predict(X=labeled_X)
        prob_val_labeled_list, dense_val_labeled_list = prob_val_labeled_list.cpu().numpy(), dense_val_labeled_list.cpu().numpy()
    print("Getting eqs' prob and feature")
    prob_val_eq_list, dense_val_eq_list = predict_probs_list, predict_fea_list

    print("Select each batch's eqs based on score")
    best_abduced_list = []
    for i in tqdm(range(0, len(abduced_list), abduction_batch_size)): # Every batch eq
        dense_val_list = np.concatenate(dense_val_eq_list[i:i+abduction_batch_size])
        prob_val_list = np.concatenate(prob_val_eq_list[i:i+abduction_batch_size])
        if labeled_X is not None:
            dense_val_list = np.concatenate((dense_val_labeled_list, dense_val_list))
            prob_val_list = np.concatenate((prob_val_labeled_list, prob_val_list))
        # Compared distance for img pair
        pair_distance = pairwise_distances(dense_val_list, metric="cosine")
        if beam_width == None or abduction_batch_size==1:
            abduced_results = gen_abduced_list(([labeled_y],*abduced_list[i:i+abduction_batch_size]))
            ground_label = None#list(itertools.chain.from_iterable(ground_labels_list[i:i+abduction_batch_size]))
            _, best_abduced_batch, _, _, _, _ = select_abduced_result(pair_distance, prob_val_list, abduced_results, labeled_y, ground_label, beam_width, similar_coef)
        else: # Beam search
            abduced_batch_list = gen_abduced_list(([labeled_y], abduced_list[i]))
            out_class_sum, out_class_cnt, inner_class_sum, inner_class_cnt = None, None, None, None
            for j in range(i + 1, min(i + abduction_batch_size, len(abduced_list))): # Beam search
                abduced_results = gen_abduced_list((abduced_batch_list, abduced_list[j]))
                ground_label = None#list(itertools.chain.from_iterable(ground_labels_list[i:j+1]))
                abduced_batch_list, best_abduced_batch, out_class_sum, out_class_cnt, inner_class_sum, inner_class_cnt = select_abduced_result(pair_distance, prob_val_list, abduced_results, labeled_y, ground_label, beam_width, similar_coef = similar_coef, inner_sum_old = inner_class_sum, inner_cnt_old = inner_class_cnt, outer_sum_old = out_class_sum, outer_cnt_old = out_class_cnt)
        best_abduced_list.extend(list_split(best_abduced_batch[len(labeled_y):], abduced_list[i:i+abduction_batch_size]))
    return best_abduced_list

def gen_abduced_list(abduced_iterables):
    # Generate abduced candidates
    if len(abduced_iterables) == 2:
        a = np.array(abduced_iterables[0], dtype=np.uint8)
        b = np.array(abduced_iterables[1], dtype=np.uint8)
        left = np.repeat(a, len(b), axis=0)
        right = np.tile(b, (len(a),1))
        result = np.zeros((left.shape[0], left.shape[1]+right.shape[1]), dtype=np.uint8)
        result[:,:left.shape[1]]=left
        result[:,left.shape[1]:]=right
        #result = np.concatenate((left, right),axis=1)
        return result
    abduced_results = []
    for abduced in itertools.product(*abduced_iterables):
        abduced_results.append(list(itertools.chain.from_iterable(abduced)))
    return np.array(abduced_results, dtype=np.uint8)

if __name__ == "__main__":    
    pass

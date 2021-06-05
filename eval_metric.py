# encoding: utf-8


import torch


def span_f1(predicts,span_label_ltoken,real_span_mask_ltoken):
    '''
    :param predicts: the prediction of model
    :param span_label_ltoken: the label of span
    :param real_span_mask_ltoken: 1 for real span, and 0 for padding span.
    '''
    pred_label_idx = torch.max(predicts, dim=-1)[1] # (bs, n_span)
    pred_label_mask = (pred_label_idx!=0)  # (bs, n_span)
    all_correct = pred_label_idx == span_label_ltoken
    all_correct = all_correct*pred_label_mask*real_span_mask_ltoken.bool()
    correct_pred = torch.sum(all_correct)
    total_pred = torch.sum(pred_label_idx!=0 )
    total_golden = torch.sum(span_label_ltoken!=0)

    return torch.stack([correct_pred, total_pred, total_golden])



def span_f1_prune(all_span_idxs,predicts,span_label_ltoken,real_span_mask_ltoken):
    '''
    :param all_span_idxs: the positon of span;
    :param predicts: the prediction of model;
    :param span_label_ltoken: the label of the span.  SHAPE: (batch_size,n_span)
    :param real_span_mask_ltoken: 1 for real span, and 0 for padding span.
    '''
    pred_label_idx = torch.max(predicts, dim=-1)[1] # (bs, n_span)
    span_probs = predicts.tolist()
    nonO_idxs2labs, nonO_kidxs_all, pred_label_idx_new = get_pruning_predIdxs(pred_label_idx, all_span_idxs, span_probs)
    pred_label_idx = pred_label_idx_new.cuda()
    pred_label_mask = (pred_label_idx!=0)  # (bs, n_span)
    all_correct = pred_label_idx == span_label_ltoken
    all_correct = all_correct*pred_label_mask*real_span_mask_ltoken.bool()
    correct_pred = torch.sum(all_correct)
    total_pred = torch.sum(pred_label_idx!=0 )
    total_golden = torch.sum(span_label_ltoken!=0)

    return torch.stack([correct_pred, total_pred, total_golden]),pred_label_idx

def get_predict(args,all_span_word, words,predicts,span_label_ltoken,all_span_idxs):
    '''
    :param all_span_word: tokens for a span;
    :param words: token in setence-level;
    :param predicts: the prediction of model;
    :param span_label_ltoken: the label for span;
    :param all_span_idxs: the position for span;
    '''
    pred_label_idx = torch.max(predicts, dim=-1)[1] # (bs, n_span)
    # for context
    idx2label = {}
    label2idx_list = args.label2idx_list
    for labidx in label2idx_list:
        lab, idx = labidx
        idx2label[int(idx)] = lab

    batch_preds = []
    for span_idxs,word,ws,lps,lts in zip(all_span_idxs,words,all_span_word,pred_label_idx,span_label_ltoken):
        text = ' '.join(word) +"\t"
        for sid,w,lp,lt in zip(span_idxs,ws,lps,lts):
            if lp !=0 or lt!=0:
                plabel = idx2label[int(lp.item())]
                tlabel = idx2label[int(lt.item())]
                sidx, eidx = sid
                ctext = ' '.join(w)+ ':: '+str(int(sidx))+','+str(int(eidx+1))  +':: '+tlabel +':: '+plabel +'\t'
                text +=ctext
        batch_preds.append(text)
    return batch_preds


def get_predict_prune(args,all_span_word, words,predicts_new,span_label_ltoken,all_span_idxs):
    '''
    :param all_span_word: tokens for a span;
    :param words: token in setence-level;
    :param predicts_new: the prediction of model;
    :param span_label_ltoken: the label for span;
    :param all_span_idxs: the position for span;
    '''
    # for context
    idx2label = {}
    label2idx_list = args.label2idx_list
    for labidx in label2idx_list:
        lab, idx = labidx
        idx2label[int(idx)] = lab

    batch_preds = []
    for span_idxs,word,ws,lps,lts in zip(all_span_idxs,words,all_span_word,predicts_new,span_label_ltoken):
        text = ' '.join(word) +"\t"
        for sid,w,lp,lt in zip(span_idxs,ws,lps,lts):
            if lp !=0 or lt!=0:
                plabel = idx2label[int(lp.item())]
                tlabel = idx2label[int(lt.item())]
                sidx, eidx = sid
                ctext = ' '.join(w)+ ':: '+str(int(sidx))+','+str(int(eidx+1))  +':: '+tlabel +':: '+plabel +'\t'
                text +=ctext
        batch_preds.append(text)
    return batch_preds


def has_overlapping(idx1, idx2):
    overlapping = True
    if (idx1[0] > idx2[1] or idx2[0] > idx1[1]):
        overlapping = False
    return overlapping

def clean_overlapping_span(idxs_list,nonO_idxs2prob):
    kidxs = []
    didxs = []
    for i in range(len(idxs_list)-1):
        idx1 = idxs_list[i]

        kidx = idx1
        kidx1 = True
        for j in range(i+1,len(idxs_list)):
            idx2 = idxs_list[j]
            isoverlapp = has_overlapping(idx1, idx2)




            if isoverlapp:
                prob1 = nonO_idxs2prob[idx1]
                prob2 = nonO_idxs2prob[idx2]

                if prob1 < prob2:
                    kidx1 = False
                    didxs.append(kidx1)
                elif prob1 == prob2:
                    len1= idx1[1] - idx1[0]+1
                    len2 = idx1[1] - idx1[0] + 1
                    if len1<len2:
                        kidx1 = False
                        didxs.append(kidx1)

        if kidx1:
            flag=True
            for idx in kidxs:
                isoverlap= has_overlapping(idx1,idx)
                if isoverlap:
                    flag=False
                    prob1 = nonO_idxs2prob[idx1]
                    prob2 = nonO_idxs2prob[idx]
                    if prob1>prob2: # del the keept idex
                        kidxs.remove(idx)
                        kidxs.append(idx1)
                    break
            if flag==True:
                kidxs.append(idx1)

    if len(didxs)==0:
        kidxs.append(idxs_list[-1])
    else:
        if idxs_list[-1] not in didxs:
            kidxs.append(idxs_list[-1])

    return kidxs

def get_pruning_predIdxs(pred_label_idx, all_span_idxs,span_probs):
    nonO_kidxs_all = []
    nonO_idxs2labs = []
    # begin{Constraint the span that was predicted can not be overlapping.}
    for i, (bs, idxs) in enumerate(zip(pred_label_idx, all_span_idxs)):
        # collect the span indexs that non-O
        nonO_idxs2lab = {}
        nonO_idxs2prob = {}
        nonO_idxs = []
        for j, (plb, idx) in enumerate(zip(bs, idxs)):
            plb = int(plb.item())
            if plb != 0:  # only consider the non-O label span...
                nonO_idxs2lab[idx] = plb
                nonO_idxs2prob[idx] = span_probs[i][j][plb]
                nonO_idxs.append(idx)

        nonO_idxs2labs.append(nonO_idxs2lab)
        if len(nonO_idxs) != 0:
            nonO_kidxs = clean_overlapping_span(nonO_idxs, nonO_idxs2prob)
        else:
            nonO_kidxs = []
        nonO_kidxs_all.append(nonO_kidxs)

    pred_label_idx_new = []
    n_span = pred_label_idx.size(1)
    for i, (bs, idxs) in enumerate(zip(pred_label_idx, all_span_idxs)):
        pred_label_idx_new1 = []
        for j, (plb, idx) in enumerate(zip(bs, idxs)):
            nlb_id = 0
            if idx in nonO_kidxs_all[i]:
                nlb_id = plb
            pred_label_idx_new1.append(nlb_id)
        while len(pred_label_idx_new1) <n_span:
            pred_label_idx_new1.append(0)

        pred_label_idx_new.append(pred_label_idx_new1)
    pred_label_idx_new = torch.LongTensor(pred_label_idx_new)
    return nonO_idxs2labs,nonO_kidxs_all,pred_label_idx_new

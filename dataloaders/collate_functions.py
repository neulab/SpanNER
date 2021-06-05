# encoding: utf-8

import torch
from typing import List


def collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
             tokens,type_ids,all_span_idxs_ltoken,morph_idxs, ...
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """

    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    max_num_span = max(x[3].shape[0] for x in batch)
    output = []

    for field_idx in range(2):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    # begin{for the pad_all_span_idxs_ltoken... }
    pad_all_span_idxs_ltoken = []
    for i in range(batch_size):
        sma = []
        for j in range(max_num_span):
            sma.append((0,0))
        pad_all_span_idxs_ltoken.append(sma)
    pad_all_span_idxs_ltoken = torch.Tensor(pad_all_span_idxs_ltoken)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][2]
        pad_all_span_idxs_ltoken[sample_idx, : data.shape[0],:] = data
    output.append(pad_all_span_idxs_ltoken)
    # end{for the pad_all_span_idxs_ltoken... }


    # begin{for the morph feature... morph_idxs}
    pad_morph_len = len(batch[0][3][0])
    pad_morph = [0 for i in range(pad_morph_len)]
    pad_morph_idxs = []
    for i in range(batch_size):
        sma = []
        for j in range(max_num_span):
            sma.append(pad_morph)
        pad_morph_idxs.append(sma)
    pad_morph_idxs = torch.LongTensor(pad_morph_idxs)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][3]
        pad_morph_idxs[sample_idx, : data.shape[0], :] = data
    output.append(pad_morph_idxs)
    # end{for the morph feature... morph_idxs}


    for field_idx in [4,5,6,7]:
        pad_output = torch.full([batch_size, max_num_span], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    words = []
    for sample_idx in range(batch_size):
        words.append(batch[sample_idx][8])
    output.append(words)


    all_span_word = []
    for sample_idx in range(batch_size):
        all_span_word.append(batch[sample_idx][9])
    output.append(all_span_word)

    all_span_idxs = []
    for sample_idx in range(batch_size):
        all_span_idxs.append(batch[sample_idx][10])
    output.append(all_span_idxs)


    return output

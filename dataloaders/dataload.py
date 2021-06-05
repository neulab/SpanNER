# encoding: utf-8


import json
import torch
from tokenizers import BertWordPieceTokenizer,ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing,BertProcessing
from torch.utils.data import Dataset
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans



class BERTNERDataset(Dataset):
	"""
	Args:
		json_path: path to spanner style json
		tokenizer: BertTokenizer
		max_length: int, max length of query+context
		possible_only: if True, only use possible samples that contain answer for the query/context

	"""

	def __init__(self, args,json_path, tokenizer: BertWordPieceTokenizer, max_length: int = 128, possible_only=False,
				 pad_to_maxlen=False):
		self.all_data = json.load(open(json_path, encoding="utf-8"))
		self.tokenzier = tokenizer
		self.max_length = max_length
		self.possible_only = possible_only
		if self.possible_only:
			self.all_data = [
				x for x in self.all_data if x["start_position"]
			]
		self.pad_to_maxlen = pad_to_maxlen

		self.args = args

		self.max_spanLen = self.args.max_spanLen
		minus = int((self.max_spanLen + 1) * self.max_spanLen / 2)
		self.max_num_span = self.max_length * self.max_spanLen - minus
		self.dataname = self.args.dataname
		self.spancase2idx_dic = {}

	def __len__(self):
		return len(self.all_data)

	def __getitem__(self, item):
		"""
		Args:
			item: int, idx
		Returns:
			tokens: tokens of query + context, [seq_len]
			token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
			start_labels: start labels of NER in tokens, [seq_len]
			end_labels: end labels of NER in tokens, [seq_len]
			label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
			match_labels: match labels, [seq_len, seq_len]
			sample_idx: sample id
			label_idx: label id

		"""
		cls_tok = "[CLS]"
		sep_tok = "[SEP]"

		# begin{get the label2idx dictionary}
		label2idx = {}
		label2idx_list = self.args.label2idx_list
		for labidx in label2idx_list:
			lab, idx = labidx
			label2idx[lab] = int(idx)
		# end{get the label2idx dictionary}

		# begin{get the morph2idx dictionary}
		morph2idx = {}
		morph2idx_list = self.args.morph2idx_list
		for morphidx in morph2idx_list:
			morph, idx = morphidx
			morph2idx[morph] = int(idx)
		# end{get the morph2idx dictionary}

		data = self.all_data[item]
		tokenizer = self.tokenzier

		context = data["context"].strip()
		if '\u200b' in context:
			context = context.replace('\u200b', '')
		elif '\ufeff' in context:
			context = context.replace('\ufeff', '')
		elif '  ' in context:
			context = context.replace('  ', ' ')

		span_idxLab = data["span_posLabel"]

		sidxs = []
		eidxs = []
		for seidx, label in span_idxLab.items():
			sidx, eidx = seidx.split(';')
			sidxs.append(int(sidx))
			eidxs.append(int(eidx))

		# add space offsets
		words = context.split()

		# convert the span position into the character index, space is also a position.
		pos_span_idxs = []
		for sidx, eidx in zip(sidxs, eidxs):
			pos_span_idxs.append((sidx, eidx))

		# all span (sidx, eidx)
		all_span_idxs = enumerate_spans(context.split(), offset=0, max_span_width=self.args.max_spanLen)

		# begin{compute the span weight}
		all_span_weights = []
		for span_idx in all_span_idxs:
			weight = self.args.neg_span_weight
			if span_idx in pos_span_idxs:
				weight=1.0
			all_span_weights.append(weight)
		# end{compute the span weight}

		all_span_lens = []
		for idxs in all_span_idxs:
			sid, eid = idxs
			slen = eid - sid + 1
			all_span_lens.append(slen)


		morph_idxs = self.case_feature_tokenLevel(morph2idx, all_span_idxs, words,self.args.max_spanLen)


		context_tokens = tokenizer.encode(context, add_special_tokens=True)
		tokens = context_tokens.ids  # subword index
		type_ids = context_tokens.type_ids  # the split of two sentence on the subword-level, 0 for first sent, 1 for the second sent
		offsets = context_tokens.offsets  # the subword's start-index and end-idx of the character-level.

		all_span_idxs_ltoken, all_span_word, all_span_idxs_new_label = self.convert2tokenIdx(words, tokens, type_ids,
																							 offsets, all_span_idxs,
																							 span_idxLab)
		span_label_ltoken = []
		for seidx_str, label in all_span_idxs_new_label.items():
			span_label_ltoken.append(label2idx[label])


		'''
		an example of tokens, type_ids, and offsets value.
		inputs: 
			query = "you are beautiful ."
			context = 'i love you .'

		outputs:
			tokens:  [101, 2017, 2024, 3376, 1012, 102, 1045, 2293, 2017, 1012, 102]
			type_ids:  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
			offsets:  [(0, 0), (0, 3), (4, 7), (8, 17), (18, 19), (0, 0), (0, 1), (2, 6), (7, 10), (11, 12), (0, 0)]
			query_context_tokens.tokens: ['[CLS]', 'you', 'are', 'beautiful', '.', '[SEP]', 'i', 'love', 'you', '.', '[SEP]']
			query_context_tokens.words:  [None, 0, 1, 2, 3, None, 0, 1, 2, 3, None]
		'''

		# # the max-end-index should not exceed the max-length.
		# all_span_idxs_ltoken

		# return  tokens, type_ids, all_span_idxs_ltoken, pos_span_mask_ltoken
		tokens = tokens[: self.max_length]
		type_ids = type_ids[: self.max_length]
		all_span_idxs_ltoken = all_span_idxs_ltoken[:self.max_num_span]
		span_label_ltoken = span_label_ltoken[:self.max_num_span]
		all_span_lens = all_span_lens[:self.max_num_span]
		morph_idxs = morph_idxs[:self.max_num_span]
		all_span_weights = all_span_weights[:self.max_num_span]

		# make sure last token is [SEP]
		sep_token = tokenizer.token_to_id(sep_tok)
		if tokens[-1] != sep_token:
			assert len(tokens) == self.max_length
			tokens = tokens[:-1] + [sep_token]

		# padding to the max length.
		import numpy as np
		real_span_mask_ltoken = np.ones_like(span_label_ltoken)
		if self.pad_to_maxlen:
			tokens = self.pad(tokens, 0)
			type_ids = self.pad(type_ids, 1)
			all_span_idxs_ltoken = self.pad(all_span_idxs_ltoken, value=(0, 0), max_length=self.max_num_span)
			real_span_mask_ltoken = self.pad(real_span_mask_ltoken, value=0, max_length=self.max_num_span)
			span_label_ltoken = self.pad(span_label_ltoken, value=0, max_length=self.max_num_span)
			all_span_lens = self.pad(all_span_lens, value=0, max_length=self.max_num_span)
			morph_idxs = self.pad(morph_idxs, value=0, max_length=self.max_num_span)
			all_span_weights = self.pad(all_span_weights, value=0, max_length=self.max_num_span)

		tokens = torch.LongTensor(tokens)
		type_ids = torch.LongTensor(type_ids)  # use to split the first and second sentence.
		all_span_idxs_ltoken = torch.LongTensor(all_span_idxs_ltoken)
		real_span_mask_ltoken = torch.LongTensor(real_span_mask_ltoken)
		span_label_ltoken = torch.LongTensor(span_label_ltoken)
		all_span_lens = torch.LongTensor(all_span_lens)
		morph_idxs = torch.LongTensor(morph_idxs)
		all_span_weights = torch.Tensor(all_span_weights)

		min_idx = np.max(np.array(all_span_idxs_ltoken))


		return [
			tokens,
			type_ids,  # use to split the first and second sentence.
			all_span_idxs_ltoken,
			morph_idxs,
			span_label_ltoken,
			all_span_lens,
			all_span_weights,
			real_span_mask_ltoken,
			words,
			all_span_word,
			all_span_idxs,
		]


	def case_feature_tokenLevel(self, morph2idx, span_idxs, words, max_spanlen):
		'''
		this function use to characterize the capitalization feature.
		:return:
		'''
		caseidxs = []

		for idxs in span_idxs:
			sid, eid = idxs
			span_word = words[sid:eid + 1]
			caseidx1 = [0 for _ in range(max_spanlen)]
			for j,token in enumerate(span_word):
				tfeat = ''
				if token.isupper():
					tfeat = 'isupper'
				elif token.islower():
					tfeat = 'islower'
				elif token.istitle():
					tfeat = 'istitle'
				elif token.isdigit():
					tfeat = 'isdigit'
				else:
					tfeat = 'other'
				caseidx1[j] =morph2idx[tfeat]
			caseidxs.append(caseidx1)

		return caseidxs


	def case_feature_spanLevel(self,spancase2idx_dic,span_idxs,words):
		'''
		this function use to characterize the capitalization feature.
		:return:
		'''

		case2idx = {'isupper': 0, 'islower': 1, 'istitle': 2, 'isdigit': 3, 'other': 4}

		caseidx = []
		for idxs in span_idxs:
			sid, eid = idxs
			span_word = words[sid:eid+1]
			caseidx1 = []
			for token in span_word:
				tfeat = ''
				if token.isupper():
					tfeat='isupper'
				elif token.islower():
					tfeat = 'islower'
				elif token.istitle():
					tfeat = 'istitle'
				elif token.isdigit():
					tfeat = 'isdigit'
				else:
					tfeat = 'other'
				caseidx1.append(tfeat)

			caseidx1_str = ' '.join(caseidx1)
			if caseidx1_str not in spancase2idx_dic:
				spancase2idx_dic[caseidx1_str] = len(spancase2idx_dic)+1
			caseidx.append(spancase2idx_dic[caseidx1_str])

		return caseidx,spancase2idx_dic








	def pad(self, lst, value=None, max_length=None):
		max_length = max_length or self.max_length
		while len(lst) < max_length:
			lst.append(value)
		return lst

	def convert2tokenIdx(self, words, tokens, type_ids, offsets, span_idxs, span_idxLab):
		# convert the all the span_idxs from word-level to token-level
		max_length = self.max_length
		sidxs = [x1 + sum([len(w) for w in words[:x1]]) for (x1, x2) in span_idxs]
		eidxs = [x2 + sum([len(w) for w in words[:x2 + 1]]) for (x1, x2) in span_idxs]

		span_idxs_new_label = {}
		for ns, ne, ose in zip(sidxs, eidxs, span_idxs):
			os, oe = ose
			oes_str = "{};{}".format(os, oe)
			nes_str = "{};{}".format(ns, ne)
			if oes_str in span_idxLab:
				label = span_idxLab[oes_str]
				span_idxs_new_label[nes_str] = label
			else:
				span_idxs_new_label[nes_str] = 'O'

		origin_offset2token_sidx = {}
		origin_offset2token_eidx = {}
		for token_idx in range(len(tokens)):
			# skip query tokens
			token_start, token_end = offsets[token_idx]

			# skip [CLS] or [SEP]
			if token_start == token_end == 0:
				continue
			origin_offset2token_sidx[token_start] = token_idx
			origin_offset2token_eidx[token_end] = token_idx

		# convert the position from character-level to token-level.
		span_new_sidxs = []
		span_new_eidxs = []
		n_span_keep = 0

		for start, end in zip(sidxs, eidxs):
			if origin_offset2token_eidx[end] > max_length - 1 or origin_offset2token_sidx[
				start] > max_length - 1:
				continue
			span_new_sidxs.append(origin_offset2token_sidx[start])
			span_new_eidxs.append(origin_offset2token_eidx[end])
			n_span_keep += 1

		all_span_word = []
		for (sidx, eidx) in span_idxs:
			all_span_word.append(words[sidx:eidx + 1])
		all_span_word = all_span_word[:n_span_keep + 1]

		span_idxs_ltoken = []
		for sidx, eidx in zip(span_new_sidxs, span_new_eidxs):
			span_idxs_ltoken.append((sidx, eidx))


		return span_idxs_ltoken, all_span_word, span_idxs_new_label

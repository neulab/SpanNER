# -*- coding: utf-8 -*
import codecs
import numpy as np
from collections import Counter
import os
import pickle
import random
def get_chunk_type(tok):
	"""
	Args:
		tok: id of token, ex 4
		idx_to_tag: dictionary {4: "B-PER", ...}
	Returns:
		tuple: "B", "PER"
	"""
	# tag_name = idx_to_tag[tok]
	tag_class = tok.split('-')[0]
	# tag_type = tok.split('-')[-1]
	tag_type = '-'.join(tok.split('-')[1:])
	return tag_class, tag_type

def get_chunks(seq):
	"""
	tags:dic{'per':1,....}
	Args:
		seq: [4, 4, 0, 0, ...] sequence of labels
		tags: dict["O"] = 4
	Returns:
		list of (chunk_type, chunk_start, chunk_end)

	Example:
		seq = [4, 5, 0, 3]
		tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
		result = [("PER", 0, 2), ("LOC", 3, 4)]
	"""
	default = 'O'
	# idx_to_tag = {idx: tag for tag, idx in tags.items()}
	chunks = []
	chunk_type, chunk_start = None, None
	for i, tok in enumerate(seq):
		#End of a chunk 1
		if tok == default and chunk_type is not None:
			# Add a chunk.
			chunk = (chunk_type, chunk_start, i)
			chunks.append(chunk)
			chunk_type, chunk_start = None, None

		# End of a chunk + start of a chunk!
		elif tok != default:
			tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
			if chunk_type is None:
				chunk_type, chunk_start = tok_chunk_type, i
			elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
				chunk = (chunk_type, chunk_start, i)
				chunks.append(chunk)
				chunk_type, chunk_start = tok_chunk_type, i
		else:
			pass
	# end condition
	if chunk_type is not None:
		chunk = (chunk_type, chunk_start, len(seq))
		chunks.append(chunk)

	return chunks


def get_chunks_onesent(seq, sentid):
	"""
	tags:dic{'per':1,....}
	Args:
		seq: [4, 4, 0, 0, ...] sequence of labels
		tags: dict["O"] = 4
	Returns:
		list of (chunk_type, chunk_start, chunk_end)

	Example:
		seq = [4, 5, 0, 3]
		tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
		result = [("PER", 0, 2), ("LOC", 3, 4)]
	"""
	default = 'O'
	# idx_to_tag = {idx: tag for tag, idx in tags.items()}
	chunks = []
	chunk_type, chunk_start = None, None
	for i, tok in enumerate(seq):
		#End of a chunk 1
		if tok == default and chunk_type is not None:
			# Add a chunk.
			chunk = (chunk_type, chunk_start, i, sentid)
			chunks.append(chunk)
			chunk_type, chunk_start = None, None

		# End of a chunk + start of a chunk!
		elif tok != default:
			tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
			if chunk_type is None:
				chunk_type, chunk_start = tok_chunk_type, i
			elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
				chunk = (chunk_type, chunk_start, i, sentid)
				chunks.append(chunk)
				chunk_type, chunk_start = tok_chunk_type, i
		else:
			pass
	# end condition
	if chunk_type is not None:
		chunk = (chunk_type, chunk_start, len(seq), sentid)
		chunks.append(chunk)

	return chunks


def evaluate_chunk_level(pred_chunks,true_chunks):
	# print("type pred_chunks: ", type(pred_chunks))
	# print("evaluate_chunk_level!")
	# print("pred_chunks[0:5]: ", pred_chunks[0:5])
	# print("true_chunks[0:5]: ", true_chunks[0:5])
	correct_preds, total_correct, total_preds = 0., 0., 0.
	correct_preds += len(set(true_chunks) & set(pred_chunks))
	total_preds += len(pred_chunks)
	total_correct += len(true_chunks)
	#


	p = correct_preds / total_preds if correct_preds > 0 else 0
	r = correct_preds / total_correct if correct_preds > 0 else 0
	f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
	# acc = np.mean(accs)
	cp = correct_preds
	tp = total_preds

	# print("f1, p, r: ", f1, p, r)
	return f1, p, r,correct_preds, total_preds, total_correct



def evaluate_each_class(pred_chunks,true_chunks, class_type):
	# class_type:PER or LOC or ORG
	# pred_chunks,true_chunks are one-dim.
	pred_chunk_class = []
	for pchunk in pred_chunks:
		if pchunk[0] == class_type:
			pred_chunk_class.append(pchunk)

	true_chunk_class = []
	for tchunk in pred_chunks:
		if tchunk[0] == class_type:
			true_chunk_class.append(tchunk)
	pred_chunk_class = set(pred_chunk_class)
	true_chunk_class = set(true_chunk_class)



	correct_preds = len((pred_chunk_class & set(true_chunks) ))
	total_preds = len(pred_chunk_class)
	total_correct = len(true_chunk_class)
	# print("type: ", class_type)
	# print("correct_preds, total_preds, total_correct: ", correct_preds,total_preds,total_correct)

	p = correct_preds / total_preds if correct_preds > 0 else 0
	r = correct_preds / total_correct if correct_preds > 0 else 0
	f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

	return f1, p, r,correct_preds,total_preds,total_correct

def evaluate_ByCategory(pred_chunks,true_chunks, class_types):
	# class_type:PER or LOC or ORG
	# pred_chunks,true_chunks are one-dim.
	class2f1_dic ={}
	for class_type in class_types:
		pred_chunk_class = []
		for pchunk in pred_chunks:
			if pchunk[0] == class_type:
				pred_chunk_class.append(pchunk)

		true_chunk_class = []
		for tchunk in pred_chunks:
			if tchunk[0] == class_type:
				true_chunk_class.append(tchunk)
		pred_chunk_class = set(pred_chunk_class)
		true_chunk_class = set(true_chunk_class)



		correct_preds = len((pred_chunk_class & set(true_chunks) ))
		total_preds = len(pred_chunk_class)
		total_correct = len(true_chunk_class)
		# print("type: ", class_type)
		# print("correct_preds, total_preds, total_correct: ", correct_preds,total_preds,total_correct)

		p = correct_preds / total_preds if correct_preds > 0 else 0
		r = correct_preds / total_correct if correct_preds > 0 else 0
		f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

		class2f1_dic[class_type] = f1

	return class2f1_dic
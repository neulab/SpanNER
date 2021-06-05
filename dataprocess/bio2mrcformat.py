import numpy as np
import json
import codecs
from collections import Counter
import os


def get_chunk_type(tok):
	"""
	Args:
		tok: id of token, ex 4
		idx_to_tag: dictionary {4: "B-PER", ...}
	Returns:
		tuple: "B", "PER"
	"""
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


def keep_spanPred_data(dataname,fpath_bio,column_no,delimiter):
	word_seqs, trueTag_seqs, word_seqs_sent, trueTag_seqs_sent = read_data(
		dataname, fpath_bio, column_no=column_no,delimiter =delimiter)  # column_no=3 for ontonotes5.0

	all_labs = []
	for tag in trueTag_seqs:
		if tag!='O':
			tags = tag.split('-')
			if len(tags)>2:
				lab = '-'.join(tags[1:])
			else:
				pre,lab  =tag.split('-')
			all_labs.append(lab)

	counter = Counter(all_labs).most_common()

	tag_dic = {"O":0}
	for i,elem in enumerate(counter):
		tag, c = elem
		tag_dic[tag] =i+1
	print(tag_dic)


	all_datas = []
	for i, (tokens,labs) in enumerate(zip(word_seqs_sent, trueTag_seqs_sent)):
		chunks = get_chunks(labs)
		context = ' '.join(tokens)
		if "[emoji]  " in context:
			print('context: ',context)
			print('tokens: ', tokens)

		pos = {}
		for chunk in chunks:
			lab, sidx, eidx = chunk
			key1 = str(sidx) +';'+str(eidx-1)
			pos[key1] = lab

		one_samp = {
			"context": context,
			"span_posLabel": pos
		}

		all_datas.append(one_samp)


	return all_datas


def read_data(corpus_type, fn, column_no=-1, delimiter =' '):
	print('corpus_type',corpus_type)
	word_sequences = list()
	tag_sequences = list()
	total_word_sequences = list()
	total_tag_sequences = list()
	with codecs.open(fn, 'r', 'utf-8') as f:
		lines = f.readlines()
	curr_words = list()
	curr_tags = list()
	for k in range(len(lines)):
		line = lines[k].strip()
		if "❤ ️" in line:
			line =line.replace("❤ ️️", "[emoji]")

		if len(line) == 0: # new sentence or new document
			if len(curr_words) > 0:
				word_sequences.append(curr_words)
				tag_sequences.append(curr_tags)
				curr_words = list()
				curr_tags = list()
			continue

		strings = line.split(delimiter)
		word = strings[0].strip()
		tag = strings[column_no].strip()  # be default, we take the last tag
		if corpus_type=='ptb2':
			tag='B-'+tag
		if word =='❤ ':
			word="[emoji]"
		word =word.strip()
		curr_words.append(word)
		curr_tags.append(tag)
		total_word_sequences.append(word)
		total_tag_sequences.append(tag)
		if k == len(lines) - 1:
			word_sequences.append(curr_words)
			tag_sequences.append(curr_tags)

	return total_word_sequences,total_tag_sequences,word_sequences,tag_sequences







if __name__ == '__main__':
	dataname = 'conll03'

	suffixs = ['train', 'dev', 'test']
	column_no = -1 # tag position
	delimiter = ' '
	if dataname =='ontonote5':
		column_no = 3
	elif 'wnut' in dataname:
		column_no =1
		delimiter = '\t'

	# convsert conll-2003 to spanner format
	fpath_bio1 = '../data/conll03_bio'
	dump_path = "../data/conll03v2/"

	if not os.path.exists(dump_path):
		os.makedirs(dump_path)

	for suffix in suffixs:
		fpath_bio = fpath_bio1 +'/' +suffix+'.txt'

		all_data = keep_spanPred_data(dataname,fpath_bio, column_no,delimiter)
		dump_file_path = dump_path + 'spanner.' + suffix
		with open(dump_file_path, "w") as f:
			json.dump(all_data, f, sort_keys=True, ensure_ascii=False, indent=2)



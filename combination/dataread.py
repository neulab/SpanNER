# -*- coding: utf-8 -*
import codecs
import os
import pickle
from evaluate_metric import evaluate_ByCategory, get_chunks_onesent,evaluate_chunk_level



class DataReader():
	def __init__(self,dataname, file_dir, classes,fmodels, fn_stand_res):
		self.dataname = dataname
		self.file_dir = file_dir
		self.fmodels = fmodels
		self.classes = classes
		self.fn_stand_res = fn_stand_res


	def read_seqModel_data(self, fn, column_no=-1, delimiter =' '):
		# # read seq model's results
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
			if len(line) == 0 or line.startswith('-DOCSTART-'): # new sentence or new document
				if len(curr_words) > 0:
					word_sequences.append(curr_words)
					tag_sequences.append(curr_tags)
					curr_words = list()
					curr_tags = list()
				continue

			strings = line.split(delimiter)
			word = strings[0].strip()
			tag = strings[column_no].strip()  # be default, we take the last tag
			if tag=='work' or tag=='creative-work': # for wnut17
				tag='work'
			if self.dataname=='ptb2':
				tag='B-'+tag
			curr_words.append(word)
			curr_tags.append(tag)
			total_word_sequences.append(word)
			total_tag_sequences.append(tag)
			if k == len(lines) - 1:
				word_sequences.append(curr_words)
				tag_sequences.append(curr_tags)

		return total_word_sequences,total_tag_sequences,word_sequences,tag_sequences

	# get the predict result of sequence model
	def get_seqModel_pred(self,fpath):
		test_word_sequences, test_trueTag_sequences, test_word_sequences_sent, test_trueTag_sequences_sent = self.read_seqModel_data(
			fpath, column_no=1)
		_, test_predTag_sequences, _, test_predTag_sequences_sent = self.read_seqModel_data(fpath, column_no=2)

		pchunk2label_dic = {}
		tchunks = []
		pchunks = []
		for sentid, (true_tags, pred_tags) in enumerate(
				zip(test_trueTag_sequences_sent, test_predTag_sequences_sent)):
			tchunk = get_chunks_onesent(true_tags, sentid)
			pchunk = get_chunks_onesent(pred_tags, sentid)
			tchunks += tchunk
			pchunks += pchunk

			for pchunk1 in pchunk:
				label, sid, eid, sentid = pchunk1
				pchunk2label_dic[(sid, eid, sentid)] = label

		return tchunks, pchunks, pchunk2label_dic

	# get the span predict result of span prediction model,
	def get_spanModel_pred(self,fpath):
		fread = open(fpath, 'r')
		lines = fread.readlines()
		snum = -1
		tchunks = []
		pchunks = []
		pchunk2label_dic={}

		for j, line in enumerate(lines):
			line = line.strip()
			if '-DOCSTART-' in line:
				continue
			else:
				snum += 1
				spans1 = line.split('\t')
				pchunk = []
				tchunk = []

				for i, span1 in enumerate(spans1):
					if i == 0:  # skip the sentence content.
						continue
					else:
						sp, seids, ttag, ptag = span1.split(":: ")
						sid, eid = seids.split(',')

						if ptag != 'O':
							pchunk1 = (ptag, int(sid), int(eid), snum)
							pchunk.append(pchunk1)
						if ttag !='O':
							tchunk1 = (ttag, int(sid), int(eid), snum)
							tchunk.append(tchunk1)
				tchunks += tchunk
				pchunks += pchunk

				# write the predict span into a dic...
				for pchunk1 in pchunk:
					label, sid, eid, sentid = pchunk1
					pchunk2label_dic[(sid, eid, sentid)] = label

		return tchunks, pchunks, pchunk2label_dic


	def get_tchunk2lab_dic(self,tchunks_models_onedim):
		tchunk2label_dic = {}
		unque_tchunks_models = list(set(tchunks_models_onedim))
		for tchunk in unque_tchunks_models:
			label, sid, eid, sentid = tchunk
			tck_pos = (sid, eid, sentid)
			tchunk2label_dic[tck_pos] = label
		return tchunk2label_dic


	def get_allModels_pred(self,):
		tchunks_models = []
		pchunks_models = []
		pchunk2label_models = []
		pchunks_models_onedim = []
		tchunks_models_onedim = []
		class2f1_models = []


		n_model=0
		span_modelIdx =[]
		for i,fmodel in enumerate(self.fmodels):
			print("fmodel: ", fmodel)

			n_model += 1
			fpath = os.path.join(self.file_dir, fmodel)

			if 'spanNER' in fmodel:
				tchunks, pchunks, pchunk2label_dic = self.get_spanModel_pred(fpath)
				span_modelIdx.append(i)
			else:
				tchunks, pchunks, pchunk2label_dic = self.get_seqModel_pred(fpath)
			tchunks_models.append(tchunks)
			pchunks_models.append(pchunks)
			pchunk2label_models.append(pchunk2label_dic)
			pchunks_models_onedim += pchunks
			tchunks_models_onedim += tchunks

			f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(pchunks,tchunks)
			print(f1, p, r, correct_preds, total_preds, total_correct)

			class2f1_dic = evaluate_ByCategory(pchunks,tchunks, self.classes)
			class2f1_models.append(class2f1_dic)

		self.span_modelIdx =span_modelIdx
		tchunks_unique = list(set(tchunks_models_onedim))
		tchunk2label_dic = self.get_tchunk2lab_dic(tchunks_unique)

		return tchunks_models,tchunks_unique,pchunks_models,tchunks_models_onedim,pchunks_models_onedim,pchunk2label_models,tchunk2label_dic,class2f1_models

	def get_sent_word(self):
		# choose CnnWglove_lstmCrf model as the standard test-set result-file.
		fn_data = os.path.join(self.file_dir,self.fn_stand_res)
		test_word_sequences, test_trueTag_sequences, test_word_sequences_sent, test_trueTag_sequences_sent = self.read_seqModel_data(
			fn_data, column_no=1)
		return test_word_sequences_sent,test_word_sequences

	def read_span_score(self, pref_upchunks,fn_prob):
		test_word_sequences_sent,test_word_sequences = self.get_sent_word()

		fpkl = open(fn_prob, 'rb')
		label2idx, all_predicts, all_span_words = pickle.load(fpkl)

		# delete the line contain '-DOCSTART-'
		new_all_predicts, new_all_span_words = [], []
		for idx, (all_predict, all_span_word) in enumerate(zip(all_predicts, all_span_words)):
			if ['-DOCSTART-'] in all_span_word[0]:
				continue
			else:
				new_all_predicts.append(all_predict[0].tolist())
				new_all_span_words.append(all_span_word[0])

		print("len(new_all_span_words): ", len(new_all_span_words))
		print("len(new_all_predicts): ", len(new_all_predicts))
		print("len(test_word_sequences_sent): ", len(test_word_sequences_sent))
		assert len(new_all_span_words) == len(new_all_predicts) == len(test_word_sequences_sent)

		pchunk_labPrb_dic = {}

		count_notIn_spanWords = 0
		csent_exceed = []
		for num, fchunk in enumerate(pref_upchunks):
			sidx, eidx, sid = fchunk
			span_word = test_word_sequences_sent[sid][sidx:eidx]
			span_idx = label2idx['O']
			if span_word in new_all_span_words[sid]:
				span_idx = new_all_span_words[sid].index(span_word)
			else:
				count_notIn_spanWords += 1
				csent_exceed.append(sid)
			lb_probs = new_all_predicts[sid][span_idx]

			key1 = (sidx, eidx, sid)
			if key1 not in pchunk_labPrb_dic:
				pchunk_labPrb_dic[key1] = {}
				for lb, idx in label2idx.items():
					pchunk_labPrb_dic[key1][lb] = lb_probs[idx]
		print("count_notIn_spanWords (due to the maxlen of bert, and we set its label as O): ", count_notIn_spanWords)
		print("the num of sentence that exceed the maxlen of bert: ", len(list(set(csent_exceed))))

		return pchunk_labPrb_dic


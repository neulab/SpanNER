# -*- coding: utf-8 -*

import numpy as np
import os
import pickle
from dataread import DataReader
import json


def evaluate_chunk_level(pred_chunks,true_chunks):
	correct_preds, total_correct, total_preds = 0., 0., 0.
	correct_preds += len(set(true_chunks) & set(pred_chunks))
	total_preds += len(pred_chunks)
	total_correct += len(true_chunks)
	p = correct_preds / total_preds if correct_preds > 0 else 0
	r = correct_preds / total_correct if correct_preds > 0 else 0
	f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

	return f1, p, r,correct_preds, total_preds, total_correct



class CombByVoting():
	def __init__(self, dataname, file_dir, fmodels, f1s,cmodelname,classes,fn_stand_res,fn_prob):
		self.dataname = dataname
		self.file_dir = file_dir
		self.fmodels = fmodels
		self.f1s = f1s
		self.cmodelname = cmodelname
		self.fn_prob =fn_prob
		self.classes = classes

		self.mres = DataReader(dataname, file_dir,classes,fmodels,fn_stand_res)

		self.wf1 = 1.0
		self.wscore = 0.8

	def get_unique_pchunk_labs(self):
		tchunks_models,\
		tchunks_unique, \
		pchunks_models, \
		tchunks_models_onedim, \
		pchunks_models_onedim, \
		pchunk2label_models, \
		tchunk2label_dic, \
		class2f1_models=self.mres.get_allModels_pred()
		self.tchunks_unique =tchunks_unique
		self.class2f1_models = class2f1_models
		self.tchunk2label_dic = tchunk2label_dic

		# the unique chunk that predict by the model..
		pchunks_unique= list(set(pchunks_models_onedim))

		# get the unique non-O chunk's label that are predicted by all the 10 models.
		keep_pref_upchunks = []
		pchunk_plb_ms =[]
		for pchunk in pchunks_unique:
			lab, sid, eid, sentid = pchunk
			key1 = (sid, eid, sentid)
			if key1 not in keep_pref_upchunks:
				keep_pref_upchunks.append(key1)
				plb_ms = [] # the length is the num of the models
				# the first position is the pchunk
				for i in range(len(self.f1s)):
					plb = 'O'
					if key1 in pchunk2label_models[i]:
						plb = pchunk2label_models[i][key1]
					plb_ms.append(plb)
				pchunk_plb_ms.append(plb_ms)

		# get the non-O true chunk that are not be recognized..
		for tchunk in tchunks_unique:
			if tchunk not in pchunks_unique: # it means that the tchunk are not been recognized by all the models
				plab, sid, eid, sentid = tchunk
				key1 = (sid, eid, sentid)
				if key1 not in keep_pref_upchunks:
					continue

		return pchunk_plb_ms,keep_pref_upchunks





	def best_potential(self):
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

		comb_kchunks = []
		for pchunk_plb_m,pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
			sid, eid, sentid = pref_upchunks
			key1 = (sid, eid, sentid)
			if key1 in self.tchunk2label_dic:
				klb = self.tchunk2label_dic[key1]
			elif 'O' in pchunk_plb_m:
				klb = 'O'
			else:
				klb = pchunk_plb_m[0]
			if klb !='O':
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)

		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('best_potential results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		return [f1, p, r, correct_preds, total_preds, total_correct]


	def voting_majority(self):
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

		comb_kchunks = []
		for pchunk_plb_m,pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
			lb2num_dic = {}
			for plbm in pchunk_plb_m:
				if plbm not in lb2num_dic:
					lb2num_dic[plbm] = 0.0
				lb2num_dic[plbm] += 1

			klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
			if klb !='O':
				sid, eid, sentid = pref_upchunks
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)
		comb_kchunks = list(set(comb_kchunks))
		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('majority_voting results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		kf1 = int(f1 * 10000)
		fn_save_comb_kchunks = 'comb_result/VM_combine_' + str(kf1) + '.pkl'

		pickle.dump([comb_kchunks, self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))

		return [f1, p, r, correct_preds, total_preds, total_correct]

	def voting_weightByOverallF1(self):
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

		comb_kchunks = []
		for pchunk_plb_m,pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
			lb2num_dic = {}
			for plbm,f1 in zip(pchunk_plb_m, self.f1s):
				if plbm not in lb2num_dic:
					lb2num_dic[plbm]=0.0
				lb2num_dic[plbm] += f1
			klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
			if klb != 'O':
				sid, eid, sentid = pref_upchunks
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)

		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('voting_weightByOverallF1 results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		kf1 = int(f1 * 10000)
		fn_save_comb_kchunks = 'comb_result/VOF1_combine_' + str(kf1) + '.pkl'
		pickle.dump([comb_kchunks, self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))

		return [f1, p, r, correct_preds, total_preds, total_correct]

	def voting_weightByCategotyF1(self):
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

		comb_kchunks = []
		for pchunk_plb_m, pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
			lb2num_dic = {}
			for plbm, f1, cf1_dic in zip(pchunk_plb_m, self.f1s, self.class2f1_models):
				if plbm not in lb2num_dic:
					lb2num_dic[plbm] = 0.0
				if plbm=='O':
					lb2num_dic[plbm] +=f1
				else:
					lb2num_dic[plbm] += cf1_dic[plbm]
			klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
			if klb != 'O':
				sid, eid, sentid = pref_upchunks
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)

		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('voting_weightByCategotyF1 results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		kf1 = int(f1 * 10000)
		fn_save_comb_kchunks = 'comb_result/VCF1_combine_' + str(kf1) + '.pkl'
		pickle.dump([comb_kchunks, self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))

		return [f1, p, r, correct_preds, total_preds, total_correct]

	def voting_spanPred_onlyScore(self):
		wf1 = self.wf1
		wscore = self.wscore
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

		print('self.fn_prob: ',self.fn_prob)
		pchunk_labPrb_dic = self.mres.read_span_score(keep_pref_upchunks,self.fn_prob)

		comb_kchunks = []
		for pchunk_plb_m, pref_upchunk in zip(pchunk_plb_ms, keep_pref_upchunks):
			lb2num_dic = {}
			for i,(plbm,f1) in enumerate(zip(pchunk_plb_m,self.f1s)):
				if plbm not in lb2num_dic:
					lb2num_dic[plbm] = 0.0
				score = pchunk_labPrb_dic[pref_upchunk][plbm]
				# lb2num_dic[plbm] += score+0.5*f1 # best
				lb2num_dic[plbm] += wscore*score+wf1*f1  # best

			klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
			if klb != 'O':
				sid, eid, sentid = pref_upchunk
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)
		comb_kchunks = list(set(comb_kchunks))




		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('voting_spanPred_onlyScore results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		kf1 = int(f1*10000)
		fn_save_comb_kchunks = 'comb_result/SpanNER_combine_'+str(kf1)+'.pkl'
		pickle.dump([comb_kchunks,self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))


		return [f1, p, r, correct_preds, total_preds, total_correct]


if __name__ == '__main__':
	corpus_types = ["conll03"]
	# corpus_types = ["conll03","notebn","notenw", "notebc", "notemz", "notewb", "notetc", "conll02dutch",
	# 				"conll02spanish", "wnut16", "wnut17"]  # "conll03"

	column_true_tag_test, column_pred_tag_test = 1, 2
	classes = []
	# fprob = ''
	fn_prob = ''
	fn_stand_res_test = ''
	fn_res = []
	file_dir =''
	for corpus_type in corpus_types:
		if corpus_type == "conll03":
			fn_res = [
				"conll03_CflairWnon_lstmCrf_1_test_9241.txt",
				"conll03_CbertWglove_lstmCrf_1_test_9201.txt",
				"conll03_CbertWnon_lstmCrf_1_test_9246.txt",
				"conll03_CflairWglove_lstmCrf_1_test_9302.txt",
				"conll03_CelmoWglove_lstmCrf_95803618_test_9211.txt",
				"conll03_CcnnWglove_lstmCrf_72102467_test_9088.txt",
				"conll03_CcnnWglove_cnnCrf_45725566_test_8971.txt",
				"conll03_CelmoWnon_lstmCrf_81319158_test_9199.txt",
				"conll03_CnonWrand_lstmCrf_03689925_test_7849.txt",
				"conll03_CcnnWrand_lstmCrf_43667285_test_8303.txt",
				"conll03_spanNER_generic_test_9157.txt",
				"conll03_spanNER_lenDecode_9228.txt",
			]
			fn_stand_res_test = 'conll03_CcnnWglove_lstmCrf_72102467_test_9088.txt'
			fn_stand_res_dev = 'conll03_CcnnWglove_lstmCrf_72102467_dev.txt'
			fn_prob = 'conll03_spanner_prob.pkl'
			classes = ["ORG", "PER", "LOC", "MISC"]

		file_dir = "results/" +corpus_type
		print('file_dir: ',file_dir)

		f1s = []
		fnames = []
		for fname in fn_res:
			f1 = float(fname.split('_')[-1].split('.')[0]) / 10000
			f1s.append(f1)
			fnames.append(fname)

		# sort the model by descend
		f1s, fnames = (list(t) for t in zip(*sorted(zip(f1s, fnames), reverse=True)))
		for f1, fname in zip(f1s, fnames):
			print(f1, fname)

		cfn_testss = [fnames,
				   fnames[:10],
				   fnames[:9],
				   fnames[:8],
				   fnames[:7],
				   fnames[:6],
				   fnames[:5],
				   fnames[:4],
				   fnames[:3],
				   fnames[:2],
				   fnames[2:4],
				   fnames[4:6],
				   fnames[3:6],
				  fnames[1:],
				  fnames[2:],
				  fnames[3:],
				  fnames[4:],
				  fnames[5:],
				   fnames[6:],
				   fnames[7:],
				   fnames[8:],
				   fnames[9:],
				   fnames[10:]
				   ]
		cf1ss = [f1s,
				 f1s[:10],
				f1s[:9],
				f1s[:8],
				f1s[:7],
				f1s[:6],
				f1s[:5],
				f1s[:4],
				f1s[:3],
				f1s[:2],
				 f1s[2:4],
				 f1s[4:6],
				 f1s[3:6],
				 f1s[1:],
				 f1s[2:],
				 f1s[3:],
				 f1s[4:],
				 f1s[5:],
				 f1s[6:],
				 f1s[7:],
				 f1s[8:],
				 f1s[9:],
				 f1s[10:]
				]



		model_names = []
		for fname in fnames:
			elems = fname.split('_')
			data_name = elems[0]

			if data_name in corpus_types:
				model_name = '_'.join(elems[1:3])
			else:
				print('get the model name error!')
				print()
				print()
				print()
				print(fname)
				break
			model_names.append(model_name)

			print('model_name: ',model_name)

		cmodel_names = [model_names,
				 model_names[:10],
				 model_names[:9],
				 model_names[:8],
				 model_names[:7],
				 model_names[:6],
				 model_names[:5],
				 model_names[:4],
				 model_names[:3],
				 model_names[:2],
				 model_names[2:4],
				 model_names[4:6],
				 model_names[3:6],
				 model_names[1:],
				 model_names[2:],
				 model_names[3:],
				 model_names[4:],
				 model_names[5:],
				 model_names[6:],
				 model_names[7:],
				model_names[8:],
				model_names[9:],
				model_names[10:]
				 ]


		print("cfn_testss:", cfn_testss)
		for fnames, f1s in zip(cfn_testss, cf1ss):
			print(fnames, f1s)

		result_store_dic = {}
		def result_store(dic, llist, name):
			if name not in dic:
				dic[name] = []
			dic[name].append(llist)
			return dic


		fn_prob = os.path.join(file_dir, fn_prob)
		for cfn_tests, cf1s,cmodelname in zip(cfn_testss, cf1ss,cmodel_names):
			print()
			print('cfn_tests', cfn_tests)
			mres = DataReader(corpus_type, file_dir, classes, cfn_tests, fn_stand_res_test)

			tchunks_models, tchunks_unique, pchunks_models, tchunks_models_onedim, pchunks_models_onedim, pchunk2label_models, tchunk2label_dic,class2f1_models = mres.get_allModels_pred()
			comvote = CombByVoting(corpus_type, file_dir, cfn_tests, cf1s,cmodelname,classes,fn_stand_res_test,fn_prob)

			# res = comvote.best_potential()
			# result_store_dic = result_store(result_store_dic, res, name='best_potential')

			res = comvote.voting_majority()
			result_store_dic = result_store(result_store_dic, res, name='voting_majority')

			res = comvote.voting_weightByOverallF1()
			result_store_dic = result_store(result_store_dic, res, name='voting_weightByOverallF1')

			res = comvote.voting_weightByCategotyF1()
			result_store_dic = result_store(result_store_dic, res, name='voting_weightByCategotyF1')

			res = comvote.voting_spanPred_onlyScore()
			result_store_dic = result_store(result_store_dic, res, name='voting_spanPred_onlyScore')


		# print results...
		vote_names = []
		resultss= []
		for vote_name,results in result_store_dic.items():
			print("vote_name: ", vote_name)
			kres = []
			for result in results:
				print(result[0])
				kres.append(result[0])
			print()
			# 转置
			vote_names.append(vote_name)
			resultss.append(kres)
		print()
		print('dataset: ', corpus_type)
		print(', '.join(vote_names))
		resultss1 = np.array(resultss).T

		for res1 in resultss1:
			# print(res1)
			res2 = ['%.2f'%(x*100) for x in res1]
			res2 = ', '.join(res2)


			print(res2)












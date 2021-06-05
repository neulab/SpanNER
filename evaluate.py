# encoding: utf-8


import os
from pytorch_lightning import Trainer

# from trainer_spanPred import BertLabeling # old evaluation version
# from trainer_spanPred_newEval import BertLabeling # new evaluation version
from trainer import BertNerTagger # start 0111

def evaluate(ckpt, hparams_file):
	"""main"""

	trainer = Trainer(gpus=[5], distributed_backend="dp")
	# trainer = Trainer(distributed_backend="dp")

	model = BertNerTagger.load_from_checkpoint(
		checkpoint_path=ckpt,
		hparams_file=hparams_file,
		map_location=None,
		batch_size=1,
		max_length=128,
		workers=0
	)
	trainer.test(model=model)


if __name__ == '__main__':

	root_dir1 = "/home/jlfu/SPred/train_logs"

	# datas = ['notenw','notewb','notebc','notemz','notetc','notebn','conll02dutch'] #
	# # datas = ['conll02spanish']  #
	# datas = ['wnut17']  #
	# for data in datas:
	# 	root_dir = os.path.join(root_dir1, data)
	# 	files = os.listdir(root_dir)
	# 	for file in files:
	# 		print('file:',file)
	# 		if "spanPred_bert" in file:
	# 			fmodel = os.path.join(root_dir,file)
	# 			fnames = os.listdir(fmodel)
	# 			ckmodel = ''
	# 			for fname in fnames:
	# 				if '.ckpt' in fname:
	# 					ckmodel= fname
	# 			CHECKPOINTS= os.path.join(fmodel,ckmodel)
	# 			HPARAMS = fmodel+"/lightning_logs/version_0/hparams.yaml"
	# 			evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)





	# # 0125
	# # conll03 bert-large, 9245 evaluation
	# midpath = "conll03/spanPred_bert-large-uncased_prunTrue_spLenTrue_spMorphFalse_SpWtFalse_value1_25149666_9245"
	# model_names = ["epoch=18.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ",mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" +mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

	# # 0125
	# # conll03 bert-large, 9245 evaluation
	# midpath = "wnut16/spanPred_bert-large-uncased_prunTrue_spLenFalse_spMorphFalse_SpWtFalse_value1_12635765"
	# model_names = ["epoch=11.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ",mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" +mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

	# 0129
	# # conll03 bert-large, 9252 evaluation
	# midpath = "conll03/spanPred_dev2train_bert-large-uncased_maxSpan4prunFalse_spLenTrue_spMorphFalse_SpWtFalse_value1_35462812"
	# model_names = ["epoch=18.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ",mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" +mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

	# # 0129
	# # conll03 bert-large, base 9157, prune evaluation
	# midpath = "conll03/spanPred_bert-large-uncased_maxSpan4prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_61661034"
	# model_names = ["epoch=13.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ",mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" +mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

	# # conll03 bert-large, base+len 9222, prune evaluation
	# midpath = "conll03/spanPred_bert-large-uncased_maxSpan4prunFalse_spLenTrue_spMorphFalse_SpWtFalse_value1_09854370"
	# model_names = ["epoch=13.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ", mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" + mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

	# # # 0130
	# # wnut17 bert-large, base 9157, prune evaluation
	# midpath = "wnut17/spanPred_bert-large-uncased_prunFalse_spLenTrue_spMorphFalse_SpWtFalse_value1_96521534"
	# model_names = ["epoch=26.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ", mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" + mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)
	#
	# # # 0130
	# # wnut17 bert-large, base 9157, prune evaluation
	# midpath = "wnut17/spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_09063161"
	# model_names = ["epoch=13.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ", mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" + mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

	# 0130
	# # conll03 bert-large, base 9321, prune evaluation
	# midpath = "conll03/spanPred_dev2train_bert-large-uncased_prunTrue_spLenFalse_spMorphFalse_SpWtFalse_value1_35932770_9321"
	# model_names = ["epoch=5.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ", mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" + mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

# # conll03 bert-large, base 9321, prune evaluation
# 	midpath = "conll03/spanPred_dev2train_bert-large-uncased_prunTrue_spLenTrue_spMorphTrue_SpWtFalse_value1_76851666_9318"
# 	model_names = ["epoch=14.ckpt"]
# 	for mn in model_names:
# 		print("model-name: ", mn)
# 		CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" + mn
# 		HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
# 		evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

#

	# # conll02spanish bert-large, base 0.873509, prune evaluation
	# midpath = "conll02spanish/spanPred_bert-base-multilingual-uncased_maxSpan4prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_62640970"
	# model_names = ["epoch=5_v0.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ", mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" + mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

	# 0125
	# conll03 bert-large, 9245 evaluation
	midpath = "notetc/spanPred_bert-large-uncased_prunFalse_spLenTrue_spMorphFalse_SpWtFalse_value1_52887159"
	model_names = ["epoch=9.ckpt"]
	for mn in model_names:
		print("model-name: ", mn)
		CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" + mn
		HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
		evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)




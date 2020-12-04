import torchvision.transforms as transforms

import argparse
import os
import ast

from core import Loaders, Base, train_an_epoch, testwithVer2, visualize_ranked_images
from tools import make_dirs, Logger, os_walk, time_now

from utils import *


def main(config):

	# init loaders and base
	loaders = Loaders(config)
	base = Base(config, loaders)

	# make directions
	make_dirs(base.output_path)
	make_dirs(base.save_model_path)
	make_dirs(base.save_logs_path)
	make_dirs(base.save_visualize_market_path)
	make_dirs(base.save_visualize_duke_path)

	# init logger
	logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'log.txt'))
	logger('\n'*3)
	logger(config)


	if config.mode == 'train':  # train mode

		# resume model from the resume_train_epoch
		start_train_epoch = 0

		# automatically resume model from the latest one
		if config.auto_resume_training_from_lastest_steps:
			root, _, files = os_walk(base.save_model_path)
			if len(files) > 0:
				# get indexes of saved models
				indexes = []
				for file in files:
					indexes.append(int(file.replace('.pkl', '').split('_')[-1]))
				indexes = sorted(list(set(indexes)), reverse=False)
				# resume model from the latest model
				base.resume_model(indexes[-1])
				#
				start_train_epoch = indexes[-1]
				logger('Time: {}, automatically resume training from the latest step (model {})'.format(time_now(), indexes[-1]))

		# main loop
		for current_epoch in range(start_train_epoch, config.total_train_epochs):
			# save model
			base.save_model(current_epoch)
			# train
			base.lr_scheduler.step(current_epoch)
			_, results = train_an_epoch(config, base, loaders, current_epoch)
			logger('Time: {};  Epoch: {};  {}'.format(time_now(), current_epoch, results))

			if current_epoch>=80 and current_epoch % 20 ==0:
				with Tick("Inference......"):
					print(config.train_dataset)
					# test
					if config.train_dataset=='duke':
						#testwithVer2(config, logger, base, loaders, 'duke', use_gcn=True, use_gm=True)
						with Tock('1 Phase'):
							duke_map, duke_rank = testwithVer2(config, logger, base, loaders, 'duke', use_gcn=False, use_gm=False)
							logger('Time: {},  base, Dataset: Duke  \nmAP: {} \nRank: {}'.format(time_now(), duke_map, duke_rank))
						with Tock('2 Phase'):
							duke_map, duke_rank = testwithVer2(config, logger, base, loaders, 'duke', use_gcn=True, use_gm=False)
							logger('Time: {},  base+gcn, Dataset: Duke  \nmAP: {} \nRank: {}'.format(time_now(), duke_map, duke_rank))
						with Tock('3 Phase'):
							duke_map, duke_rank = testwithVer2(config, logger, base, loaders, 'duke', use_gcn=True, use_gm=True)
							logger('Time: {},  base+gcn+gm, Dataset: Duke  \nmAP: {} \nRank: {}'.format(time_now(), duke_map, duke_rank))
						logger('')
					if config.train_dataset=='market':
						#testwithVer2(config, logger, base, loaders, 'market', use_gcn=True, use_gm=True)
						with Tock('1 Phase'):
							market_map, market_rank = testwithVer2(config, logger, base, loaders, 'market', use_gcn=False, use_gm=False)
							logger('Time: {},  base, Dataset: market_map, market_rank  \nmAP: {} \nRank: {}'.format(time_now(), market_map, market_rank))
						with Tock('2 Phase'):
							market_map, market_rank = testwithVer2(config, logger, base, loaders, 'market', use_gcn=True, use_gm=False)
							logger('Time: {},  base+gcn, Dataset: market_map, market_rank  \nmAP: {} \nRank: {}'.format(time_now(), market_map, market_rank))
						with Tock('3 Phase'):
							market_map, market_rank = testwithVer2(config, logger, base, loaders, 'market', use_gcn=True, use_gm=True)
							logger('Time: {},  base+gcn+gm, Dataset: market_map, market_rank  \nmAP: {} \nRank: {}'.format(time_now(), market_map, market_rank))
						logger('')


	elif config.mode == 'test':	# test mode
		# resume from the resume_test_epoch
		if config.resume_test_path != '' and config.resume_test_epoch != 0:
			base.resume_model_from_path(config.resume_test_path, config.resume_test_epoch)
		else:
			assert 0, 'please set resume_test_path and resume_test_epoch '
		# test
		print('It is in test mode!!!\n')
		print(config.train_dataset)
		if config.train_dataset=='duke':
			print('########## duke test ##########')
			with Tock('1 Phase'):
				duke_map, duke_rank = testwithVer2(config, logger, base, loaders, 'duke', use_gcn=False, use_gm=False)
				logger('Time: {},  base, Dataset: Duke  \nmAP: {} \nRank: {}'.format(time_now(), duke_map, duke_rank))
			with Tock('2 Phase'):
				duke_map, duke_rank = testwithVer2(config, logger, base, loaders, 'duke', use_gcn=True, use_gm=False)
				logger('Time: {},  base+gcn, Dataset: Duke  \nmAP: {} \nRank: {}'.format(time_now(), duke_map, duke_rank))
			with Tock('3 Phase'):
				duke_map, duke_rank = testwithVer2(config, logger, base, loaders, 'duke', use_gcn=True, use_gm=True)
				logger('Time: {},  base+gcn+gm, Dataset: Duke  \nmAP: {} \nRank: {}'.format(time_now(), duke_map, duke_rank))
			logger('')

		if config.train_dataset=='market':
			print('########## market test ##########')
			with Tock('1 Phase'):
				market_map, market_rank = testwithVer2(config, logger, base, loaders, 'market', use_gcn=False, use_gm=False)
				logger('Time: {},  base, Dataset: market_map, market_rank  \nmAP: {} \nRank: {}'.format(time_now(), market_map, market_rank))
			with Tock('2 Phase'):
				market_map, market_rank = testwithVer2(config, logger, base, loaders, 'market', use_gcn=True, use_gm=False)
				logger('Time: {},  base+gcn, Dataset: market_map, market_rank  \nmAP: {} \nRank: {}'.format(time_now(), market_map, market_rank))
			with Tock('3 Phase'):
				market_map, market_rank = testwithVer2(config, logger, base, loaders, 'market', use_gcn=True, use_gm=True)
				logger('Time: {},  base+gcn+gm, Dataset: market_map, market_rank  \nmAP: {} \nRank: {}'.format(time_now(), market_map, market_rank))
			logger('')
		
		if config.train_dataset=='pr':
			print('########## partial reid test ##########')
			with Tock('1 Phase'):
				pr_map, pr_rank = testwithVer2(config, logger, base, loaders, 'pr', use_gcn=False, use_gm=False)
				logger('Time: {},  base, Dataset: pr_map, pr_rank  \nmAP: {} \nRank: {}'.format(time_now(), pr_map, pr_rank))
			with Tock('2 Phase'):
				pr_map, pr_rank = testwithVer2(config, logger, base, loaders, 'pr', use_gcn=True, use_gm=False)
				logger('Time: {},  base+gcn, Dataset: pr_map, pr_rank  \nmAP: {} \nRank: {}'.format(time_now(), pr_map, pr_rank))
			with Tock('3 Phase'):
				pr_map, pr_rank = testwithVer2(config, logger, base, loaders, 'pr', use_gcn=True, use_gm=True)
				logger('Time: {},  base+gcn+gm, Dataset: pr_map, pr_rank  \nmAP: {} \nRank: {}'.format(time_now(), pr_map, pr_rank))
			logger('')
		
		if config.train_dataset=='pi':
			print('########## partial iLIDS test ##########')
			with Tock('1 Phase'):
				pi_map, pi_rank = testwithVer2(config, logger, base, loaders, 'pi', use_gcn=False, use_gm=False)
				logger('Time: {},  base, Dataset: pi_map, pi_rank  \nmAP: {} \nRank: {}'.format(time_now(), pi_map, pi_rank))
			with Tock('2 Phase'):
				pi_map, pi_rank = testwithVer2(config, logger, base, loaders, 'pi', use_gcn=True, use_gm=False)
				logger('Time: {},  base+gcn, Dataset: pi_map, pi_rank  \nmAP: {} \nRank: {}'.format(time_now(), pi_map, pi_rank))
			with Tock('3 Phase'):
				pi_map, pi_rank = testwithVer2(config, logger, base, loaders, 'pi', use_gcn=True, use_gm=True)
				logger('Time: {},  base+gcn+gm, Dataset: pi_map, pi_rank  \nmAP: {} \nRank: {}'.format(time_now(), pi_map, pi_rank))
			logger('')
		
		if config.train_dataset=='occreid':
			print('########## occ reid test ##########')
			with Tock('1 Phase'):
				occreid_map, occreid_rank = testwithVer2(config, logger, base, loaders, 'occreid', use_gcn=False, use_gm=False)
				logger('Time: {},  base, Dataset: occreid_map, occreid_rank  \nmAP: {} \nRank: {}'.format(time_now(), occreid_map, occreid_rank))
			with Tock('2 Phase'):
				occreid_map, occreid_rank = testwithVer2(config, logger, base, loaders, 'occreid', use_gcn=True, use_gm=False)
				logger('Time: {},  base+gcn, Dataset: occreid_map, occreid_rank  \nmAP: {} \nRank: {}'.format(time_now(), occreid_map, occreid_rank))
			with Tock('3 Phase'):
				occreid_map, occreid_rank = testwithVer2(config, logger, base, loaders, 'occreid', use_gcn=True, use_gm=True)
				logger('Time: {},  base+gcn+gm, Dataset: occreid_map, occreid_rank  \nmAP: {} \nRank: {}'.format(time_now(), occreid_map, occreid_rank))
			logger('')
		


	elif config.mode == 'visualize': # visualization mode
		# resume from the resume_visualize_epoch
		if config.resume_visualize_path != '' and config.resume_visualize_epoch != 0:
			base.resume_model_from_path(config.resume_visualize_path, config.resume_visualize_epoch)
			print('Time: {}, resume model from {} {}'.format(time_now(), config.resume_visualize_path, config.resume_visualize_epoch))
		# visualization
		if 'market' in config.train_dataset:
			visualize_ranked_images(config, base, loaders, 'market')
		elif 'duke' in config.train_dataset:
			visualize_ranked_images(config, base, loaders, 'duke')
		else:
			assert 0

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	#
	parser.add_argument('--cuda', type=str, default='cuda')
	parser.add_argument('--mode', type=str, default='train', help='train, test or visualize')
	parser.add_argument('--output_path', type=str, default='out/base/', help='path to save related informations')

	# dataset configuration
	parser.add_argument('--duke_path', type=str, default='./occluded/duke')
	parser.add_argument('--market_path', type=str, default='./occluded/market')
	parser.add_argument('--pr_path', type=str, default='/home/jayyoung/code/ReID/DATASET/partial_reid/')
	parser.add_argument('--pi_path', type=str, default='/home/jayyoung/code/ReID/DATASET/partial_ilids/')
	parser.add_argument('--occreid_path', type=str, default='/home/jayyoung/code/ReID/DATASET/Occluded_REID/')

	parser.add_argument('--train_dataset', type=str, default='duke', help='occluded_duke')
	parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128])
	parser.add_argument('--p', type=int, default=16, help='person count in a batch')
	parser.add_argument('--k', type=int, default=4, help='images count of a person in a batch')

	# model configuration
	parser.add_argument('--pid_num', type=int, default=702, help='702 DukeMTMC-reID, 751 market ')
	parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
	parser.add_argument('--branch_num', type=int, default=14, help='')

	# keypoints model
	parser.add_argument('--weight_global_feature', type=float, default=1.0, help='')
	parser.add_argument('--norm_scale', type=float, default=10.0, help='')

	# gcn model
	parser.add_argument('--gcn_scale', type=float, default=20.0, help='')
	parser.add_argument('--gcn_lr_scale', type=float, default=0.1, help='')

	# graph matching model
	parser.add_argument('--use_gm_after', type=int, default=20, help='')
	parser.add_argument('--gm_lr_scale', type=float, default=1.0, help='')
	parser.add_argument('--weight_p_loss', type=float, default=1.0, help='')

	# verification model
	parser.add_argument('--weight_ver_loss', type=float, default=0.1, help='')
	parser.add_argument('--ver_lr_scale', type=float, default=1.0, help='')
	parser.add_argument('--ver_topk', type=int, default=1, help='')
	parser.add_argument('--ver_alpha', type=float, default=0.5, help='')
	parser.add_argument('--ver_in_scale', type=float, default=10.0, help='')


	# train configuration
	parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70], help='milestones for the learning rate decay')
	parser.add_argument('--base_learning_rate', type=float, default=0.00035)
	parser.add_argument('--weight_decay', type=float, default=0.0005)
	parser.add_argument('--total_train_epochs', type=int, default=200)
	parser.add_argument('--auto_resume_training_from_lastest_steps', type=ast.literal_eval, default=True)
	parser.add_argument('--max_save_model_num', type=int, default=100, help='0 for max num is infinit')

	# test configuration
	parser.add_argument('--resume_test_path', type=str, default='', help=' for no resuming')
	parser.add_argument('--resume_test_epoch', type=int, default=0, help='0 for no resuming')

	# visualization configuration
	parser.add_argument('--resume_visualize_path', type=str, default='', help=' for no resuming')
	parser.add_argument('--resume_visualize_epoch', type=int, default=0, help='0 for no resuming')

	# main
	config = parser.parse_args()
	main(config)

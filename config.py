import argparse
import yaml

with open('config.yml','r') as fp:
	conf = yaml.load(fp, Loader=yaml.FullLoader)
	print(conf['ModelTraining']['model'])
	print(conf)

def get_args():
	parser = argparse.ArgumentParser()
	parser = argparse.ArgumentParser(description='PyTorch CK+ CNN Training')
	parser.add_argument('--model', type=str, default=conf['ModelTraining']['model'], help='CNN architecture')
	parser.add_argument('--dataset', type=str, default=conf['ModelTraining']['dataset'], help='dataset')
	parser.add_argument('--fold', default=conf['ModelTraining']['fold'], type=int, help='k fold number')
	parser.add_argument('--bs', default=conf['ModelTraining']['bs'], type=int, help='batch_size')
	parser.add_argument('--lr', default=conf['ModelTraining']['lr'], type=float, help='learning rate')
	parser.add_argument('--epoch', default=conf['ModelTraining']['epoch'], type=int, help='training epoch')
	parser.add_argument('--save-path', default=conf['ModelTraining']['path'], type=str)
	parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
	parser.add_argument('--attack', default=conf['AdvAttack']['attack_type'], type=str, choices=['none', 'pgd', 'fgsm'])
	parser.add_argument('--epsilon', default=conf['AdvAttack']['epsilon'], type=float)
	parser.add_argument('--alpha', default=conf['AdvAttack']['alpha'], type=float)
	parser.add_argument('--attack-iters', default=conf['AdvAttack']['attack_iters'], type=int)
	parser.add_argument('--repeat-num', default=conf['AdvTraining']['repeat_num'], type=int)
	return parser.parse_args()


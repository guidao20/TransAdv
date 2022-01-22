from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from CKplus import CKplus
from torch.autograd import Variable
from Networks import *
from adversarial_training import Adversarial_Trainings
from config import get_args
from attack_method import Adversarial_methods


def test(use_cuda, testloader, net, criterion, attack_dict):
    Test_acc = 0
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)

        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops

        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.data.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    Test_acc = 100.*correct/total


    return Test_acc

def test_adv(use_cuda, testloader, net, criterion, attack_dict):
	Test_acc = 0
	net.eval()
	PrivateTest_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(testloader):
		bs, ncrops, c, h, w = np.shape(inputs)
		inputs = inputs.view(-1, c, h, w)	
		targets = (targets.unsqueeze(1) * torch.ones((targets.shape[0], ncrops)).type_as(targets)).view(-1)

		if use_cuda:
		    inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs, volatile=True), Variable(targets)

		if attack_dict == None:
		    delta = torch.zeros_like(inputs)
		else:      
		    adversarial_attack = Adversarial_methods(inputs, targets, attack_dict['attack_iters'], attack_dict['net'], attack_dict['epsilon'], attack_dict['alpha'])   
		    delta = adversarial_attack.fgsm()

		outputs = net(inputs+delta)
		loss = criterion(outputs, targets)
		PrivateTest_loss += loss.item()
		_, predicted = torch.max(outputs, 1)
		total += targets.size(0)
		correct += predicted.eq(targets).cpu().sum()

		utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		    % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
		# Save checkpoint.
	Test_acc = 100.*correct/total
	return Test_acc


def test_avg_adv(use_cuda, testloader, net, criterion, attack_dict):
	Test_acc = 0 
	net.eval()
	PrivateTest_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(testloader):
		bs, ncrops, c, h, w = np.shape(inputs)
		inputs = inputs.view(-1, c, h, w)	
		targets1 = targets
		targets = (targets.unsqueeze(1) * torch.ones((targets.shape[0], ncrops)).type_as(targets)).view(-1)

		if use_cuda:
		    inputs, targets, targets1 = inputs.cuda(), targets.cuda(), targets1.cuda()
		inputs, targets = Variable(inputs, volatile=True), Variable(targets)

		if attack_dict == None:
		    delta = torch.zeros_like(inputs)
		else:      
		    adversarial_attack = Adversarial_methods(inputs, targets, attack_dict['attack_iters'], attack_dict['net'], attack_dict['epsilon'], attack_dict['alpha'])   
		    delta = adversarial_attack.fgsm()

		outputs = net(inputs+delta)

		outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
		loss = criterion(outputs_avg, targets1)
		PrivateTest_loss += loss.item()
		_, predicted = torch.max(outputs_avg.data, 1)
		total += targets1.size(0)
		correct += predicted.eq(targets1).cpu().sum()

		utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		    % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
		# Save checkpoint.
	Test_acc = 100.*correct/total
	return Test_acc
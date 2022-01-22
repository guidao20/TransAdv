import torch
from torch.autograd import Variable 
import utils
from attack_method import *
import torch.nn as nn
import torch.optim as optim
from time import clock
import os

class Trans_CNN(nn.Module):
    def __init__(self):
        super(Trans_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 5, padding = (2,2))
        self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 13, padding =(6,6))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out



class Adversarial_Trainings(object):
    def __init__(self,  repeat_num , trainloader, use_cuda, attack_iters, net, epsilon, alpha, learning_rate_decay_start, learning_rate_decay_every, learning_rate_decay_rate, lr, save_path):
        self.trainloader = trainloader
        self.repeat_num = repeat_num
        self.use_cuda = use_cuda
        self.attack_iters = attack_iters
        self.net = net
        self.epsilon = epsilon
        self.alpha = alpha
        self.learning_rate_decay_start = learning_rate_decay_start
        self.learning_rate_decay_every = learning_rate_decay_every
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.lr = lr
        self.save_path = save_path
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)



    def Stardard_Training(self, total_epoch):
        self.net.train()
        for epoch in range(total_epoch):
            print('\nEpoch: %d' % epoch)
            train_loss = 0
            correct = 0
            total = 0
            if epoch > self.learning_rate_decay_start and self.learning_rate_decay_start >= 0:
                frac = (epoch - self.learning_rate_decay_start) // self.learning_rate_decay_every
                decay_factor = self.learning_rate_decay_rate ** frac
                current_lr = self.lr * decay_factor
                utils.set_lr(self.optimizer, current_lr)  # set the decayed rate
            else:
                current_lr = self.lr
            print('learning_rate: %s' % str(current_lr))

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                self.optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = self.net(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                loss.backward()
                utils.clip_gradient(self.optimizer, 0.1)
                self.optimizer.step()

                train_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                utils.progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            Train_acc = 100.*correct/total
        torch.save(self.net.state_dict(), 'base_model.pt')
        return self.net


    def pgd_advTraining(self, total_epoch):
        self.net.train()
        for epoch in range(total_epoch):
            print('\nEpoch: %d' % epoch)
            train_loss = 0
            correct = 0
            total = 0

            if epoch > self.learning_rate_decay_start and self.learning_rate_decay_start >= 0:
                frac = (epoch - self.learning_rate_decay_start) // self.learning_rate_decay_every
                decay_factor = self.learning_rate_decay_rate ** frac
                current_lr = self.lr * decay_factor
                utils.set_lr(self.optimizer, current_lr)  # set the decayed rate
            else:
                current_lr = self.lr
            print('learning_rate: %s' % str(current_lr))

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                self.optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)

                delta = torch.zeros_like(inputs)

                for repeat_ in range(self.repeat_num):
                    # Generating adversarial examples
                    adversarial_attack = Adversarial_methods(inputs + delta, targets, self.attack_iters, self.net, self.epsilon, self.alpha) 
                    delta = adversarial_attack.fgsm()

                # Update network parameters
                outputs = self.net(torch.clamp(inputs + delta, 0, 1))
                loss = nn.CrossEntropyLoss()(outputs, targets)
                loss.backward()
                utils.clip_gradient(self.optimizer, 0.1)
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                utils.progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            Train_acc = 100.*correct/total
        print("The final free_fast accuracy is :", Train_acc)
        torch.save(self.net.state_dict(), 'pgd__base_model.pt')
        return self.net



    def fast_advTraining(self, total_epoch):
        self.net.train()
        for epoch in range(total_epoch):
            print('\nEpoch: %d' % epoch)
            train_loss = 0
            correct = 0
            total = 0

            if epoch > self.learning_rate_decay_start and self.learning_rate_decay_start >= 0:
                frac = (epoch - self.learning_rate_decay_start) // self.learning_rate_decay_every
                decay_factor = self.learning_rate_decay_rate ** frac
                current_lr = self.lr * decay_factor
                utils.set_lr(self.optimizer, current_lr)  # set the decayed rate
            else:
                current_lr = self.lr
            print('learning_rate: %s' % str(current_lr))

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                self.optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)


                # Generating adversarial examples
                adversarial_attack = Adversarial_methods(inputs, targets, self.attack_iters, self.net, self.epsilon, self.alpha) 
                delta = adversarial_attack.rfgsm()
                # Update network parameters
                outputs = self.net(torch.clamp(inputs + delta, 0, 1))
                loss = nn.CrossEntropyLoss()(outputs, targets)
                loss.backward()
                utils.clip_gradient(self.optimizer, 0.1)
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                utils.progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

            Train_acc = 100.*correct/total
        print("The final free_fast accuracy is :", Train_acc)
        torch.save(self.net.state_dict(), 'fast_base_model.pt')
        return self.net

    def free_advTraining(self, total_epoch):
        self.net.train()
        for epoch in range(total_epoch):
            print('\nEpoch: %d' % epoch)
            train_loss = 0
            correct = 0
            total = 0

            if epoch > self.learning_rate_decay_start and self.learning_rate_decay_start >= 0:
                frac = (epoch - self.learning_rate_decay_start) // self.learning_rate_decay_every
                decay_factor = self.learning_rate_decay_rate ** frac
                current_lr = self.lr * decay_factor
                utils.set_lr(self.optimizer, current_lr)  # set the decayed rate
            else:
                current_lr = self.lr
            print('learning_rate: %s' % str(current_lr))

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                self.optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)
                for repeat_ in range(self.repeat_num):
                    # Generating adversarial examples
                    adversarial_attack = Adversarial_methods(inputs, targets, self.attack_iters, self.net, self.epsilon, self.alpha) 
                    delta = adversarial_attack.fgsm()
                    # Update network parameters
                    outputs = self.net(torch.clamp(inputs + delta, 0, 1))
                    loss = nn.CrossEntropyLoss()(outputs, targets)
                    loss.backward()
                    utils.clip_gradient(self.optimizer, 0.1)
                    self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                utils.progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

            Train_acc = 100.*correct/total
        print("The final free_fast accuracy is :", Train_acc)
        torch.save(self.net.state_dict(), 'free_base_model.pt')
        return self.net



    def Trans_advtraining(self, total_epoch):
        self.net.train()
        Trans_Net = Trans_CNN()
        Trans_Net.load_state_dict(torch.load('Trans_net_parameters.pt'))
        for epoch in range(total_epoch):
            print('\nEpoch: %d' % epoch)
            train_loss = 0
            correct = 0
            total = 0
            if self.use_cuda:
                Trans_Net.cuda()
            loss_fn = nn.CrossEntropyLoss()
            if epoch > self.learning_rate_decay_start and self.learning_rate_decay_start >= 0:
                frac = (epoch - self.learning_rate_decay_start) // self.learning_rate_decay_every
                decay_factor = self.learning_rate_decay_rate ** frac
                current_lr = self.lr * decay_factor
                utils.set_lr(self.optimizer, current_lr)  # set the decayed rate
            else:
                current_lr = self.lr
            print('learning_rate: %s' % str(current_lr))

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                self.optimizer.zero_grad()
                # Generating adversarial example
                # Adversarial_method 引入trans_net参数, 每个攻击方法里引入一个新的攻击类例如fgsm_trans
                adversarial_attack = Trans_Adversarial_methods(inputs, targets, self.attack_iters, self.net, Trans_Net, self.epsilon, self.alpha)
                delta = adversarial_attack.fgsm()
                # adversarial_attack = Adversarial_methods(inputs, targets, self.attack_iters, self.net, self.epsilon, self.alpha)
                # delta = adversarial_attack.fgsm()
                x_adv = torch.clamp(inputs + delta, 0, 1)
                # Update network parameters
                outputs_adv = self.net(x_adv)
                loss = loss_fn(outputs_adv, targets)
                loss.backward()
                utils.clip_gradient(self.optimizer, 0.1)
                self.optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs_adv.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()
                utils.progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

            Train_acc = 100.*correct/total
        print("The final free_fast accuracy is :", Train_acc)
        torch.save(self.net.state_dict(), 'Trans_base_model.pt')
        return self.net


    def Trans_training(self, total_epoch):
        self.net.load_state_dict(torch.load('model/VGG19/Standard_Training/model.pt'))
        Trans_Net = Trans_CNN().train()
        for epoch in range(total_epoch):
            print('\nEpoch: %d' % epoch)
            Trans_optimizer = optim.SGD(Trans_Net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            if self.use_cuda:
                Trans_Net.cuda()
            loss_fn = nn.CrossEntropyLoss()
            if epoch > self.learning_rate_decay_start and self.learning_rate_decay_start >= 0:
                frac = (epoch - self.learning_rate_decay_start) // self.learning_rate_decay_every
                decay_factor = self.learning_rate_decay_rate ** frac
                current_lr = self.lr * decay_factor
                utils.set_lr(Trans_optimizer, current_lr)  # set the decayed rate
            else:
                current_lr = self.lr
            print('learning_rate: %s' % str(current_lr))

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                Trans_optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)
                # Generating adversarial example
                # Adversarial_method 引入trans_net参数, 每个攻击方法里引入一个新的攻击类例如fgsm_trans
                adversarial_attack = Trans_Adversarial_methods(inputs, targets, self.attack_iters, self.net, Trans_Net, self.epsilon, self.alpha) 
                delta = adversarial_attack.fgsm()
                x_adv = torch.clamp(inputs + delta, 0, 1)
                # Train Trans_NN
                # Update network parameters
                outputs_adv = self.net(Trans_Net(x_adv))
                outputs_clean = self.net(Trans_Net(inputs))
                loss1 = loss_fn(outputs_adv, targets)  
                loss2 = loss_fn(outputs_clean, targets)
                loss3 = torch.norm(Trans_Net(x_adv)-x_adv)
                loss =  0.5 * loss1 +  0.5 * loss2 + 0.001 * loss3
                print('loss1: {} loss2: {} loss3: {} loss: {}'.format(loss1, loss2, loss3, loss))
                loss.backward()
                utils.clip_gradient(Trans_optimizer, 0.1)
                Trans_optimizer.step()
        torch.save(Trans_Net.state_dict(), 'Trans_net_parameters.pt')



    def Double_min_max_training(self, total_epoch):
        self.net.train()
        Trans_Net = Trans_CNN()
        optimizer_trans = optim.SGD(Trans_Net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        for epoch in range(total_epoch):
            print('\nEpoch: %d' % epoch)
            train_loss = 0
            correct = 0
            total = 0

            if self.use_cuda:
                Trans_Net.cuda()
            loss_fn = nn.CrossEntropyLoss()
            if epoch > self.learning_rate_decay_start and self.learning_rate_decay_start >= 0:
                frac = (epoch - self.learning_rate_decay_start) // self.learning_rate_decay_every
                decay_factor = self.learning_rate_decay_rate ** frac
                current_lr = self.lr * decay_factor
                utils.set_lr(self.optimizer, current_lr)  # set the decayed rate
            else:
                current_lr = self.lr
            print('learning_rate: %s' % str(current_lr))

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                self.optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)

                # Generating adversarial example
                # Adversarial_method 引入trans_net参数, 每个攻击方法里引入一个新的攻击类例如fgsm_trans
                adversarial_attack = Adversarial_methods(inputs, targets, self.attack_iters, self.net, self.epsilon, self.alpha) 
                delta = adversarial_attack.fgsm()
                # Train Trans_NN
                # Update Trans network parameters
                outputs_adv = self.net(Trans_Net(torch.clamp(inputs + delta, 0, 1)))
                outputs_clean = self.net(Trans_Net(inputs))
                loss = 0.5 * loss_fn(outputs_adv, targets) + 0.5 * loss_fn(outputs_clean, targets) + 0.001 * torch.norm(delta)
                loss.backward()
                utils.clip_gradient(optimizer_trans, 0.1)
                optimizer_trans.step()

                # Train model Network
                # Update model network parameters
                adversarial_attack_ = Adversarial_methods(inputs, targets, self.attack_iters, self.net, self.epsilon, self.alpha) 
                delta_ = adversarial_attack_.fgsm() 

                outputs_adv_ = self.net(Trans_Net(torch.clamp(inputs + delta_, 0, 1)))
                loss_ = loss_fn(outputs_adv_, targets)
                loss_.backward()
                utils.clip_gradient(self.optimizer, 0.1)
                self.optimizer.step()

                train_loss += loss_.item()
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                utils.progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        torch.save(Trans_Net.state_dict(), 'Trans_net_parameters_double.pt')
        torch.save(self.net.state_dict(), 'Trans_net_base_double')
        Train_acc = 100.*correct/total
        print("The final free_fast accuracy is :", Train_acc)
        return self.net
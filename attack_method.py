import torch
import torch.nn as nn


class Adversarial_methods(object):
    def __init__(self, inputs, targets, attack_iter, net, epsilon, alpha):
        self.inputs = inputs
        self.targets = targets
        self.net = net
        self.epsilon = epsilon
        self.alpha = alpha


    def fgsm(self):
        delta = torch.zeros_like(self.inputs)
        delta.requires_grad = True
        outputs = self.net(self.inputs + delta)
        loss = nn.CrossEntropyLoss()(outputs, self.targets)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + self.alpha * torch.sign(grad), -self.epsilon, self.epsilon)
        delta.data = torch.max(torch.min(1 - self.inputs, delta.data), 0 - self.inputs)
        delta = delta.detach()
        return delta

    def rfgsm(self):
        delta = torch.zeros_like(self.inputs).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True
        outputs = self.net(self.inputs + delta)
        loss = nn.CrossEntropyLoss()(outputs, self.targets)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + self.alpha * torch.sign(grad), -self.epsilon, self.epsilon)
        delta.data = torch.max(torch.min(1 - self.inputs, delta.data), 0 - self.inputs)
        delta = delta.detach()
        return delta


    def pgd(self):
        delta = torch.zeros_like(self.inputs).uniform_(-self.epsilon, self.epsilon)
        delta.data = torch.max(torch.min(1 - self.inputs, delta.data), 0 - self.inputs)
        for _ in range(self.attack_iters):
            delta.requires_grad = True
            outputs = self.net(self.inputs + delta)
            loss = nn.CrossEntropyLoss()(outputs, self.targets)
            self.optimizer.zero_grad()
            loss.backward()
            grad = delta.grad.detach()
            I = outputs.max(1)[1] == self.targets
            delta.data[I] = torch.clamp(delta + self.alpha * torch.sign(grad), -self.epsilon, self.epsilon)[I]
            delta.data[I] = torch.max(torch.min(1 - self.inputs, delta.data), 0 - self.inputs)[I]
        delta = delta.detach()
        return delta


    def nothing(self):
        delta = torch.zeros_like(self.inputs)
        return delta



class Trans_Adversarial_methods(object):
    def __init__(self, inputs, targets, attack_iter, net, trans_net, epsilon, alpha):
        self.inputs = inputs
        self.targets = targets
        self.net = net
        self.trans_net = trans_net
        self.epsilon = epsilon
        self.alpha = alpha


    def fgsm(self):
        delta = torch.zeros_like(self.inputs)
        delta.requires_grad = True
        outputs1 = self.net(self.trans_net(self.inputs + delta))
        loss1 = nn.CrossEntropyLoss()(outputs1, self.targets)
        outputs2 = self.net(self.inputs + delta)
        loss2 = nn.CrossEntropyLoss()(outputs2, self.targets)
        loss = 0.5 * loss1 + 0.5 * loss2
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + self.alpha * torch.sign(grad), -self.epsilon, self.epsilon)
        delta.data = torch.max(torch.min(1 - self.inputs, delta.data), 0 - self.inputs)
        delta = delta.detach()
        return delta

    def rfgsm(self):
        delta = torch.zeros_like(self.inputs).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True
        outputs1 = self.net(self.trans_net(self.inputs + delta))
        loss1 = nn.CrossEntropyLoss()(outputs1, self.targets)
        outputs2 = self.net(self.inputs + delta)
        loss2 = nn.CrossEntropyLoss()(outputs2, self.targets)
        loss = 0.5 * loss1 + 0.5 * loss2
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + self.alpha * torch.sign(grad), -self.epsilon, self.epsilon)
        delta.data = torch.max(torch.min(1 - self.inputs, delta.data), 0 - self.inputs)
        return delta
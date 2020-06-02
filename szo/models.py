# -*- coding: utf-8 -*-

"""
Models module: defines network architecture
"""

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.utils.convert_parameters import parameters_to_vector as to_vector

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class FullyConnectedNN(torch.nn.Module):
    def __init__(self, input_dim=784, output_dim=10, hidden_dim_one=300, hidden_dim_two=100,
                 seed=None, class_names=None, norm='batch', norm_affine = False):
        super(FullyConnectedNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim_one, bias=True)
        self.fc2 = nn.Linear(hidden_dim_one, hidden_dim_two, bias=True)
        self.fc3 = nn.Linear(hidden_dim_two, output_dim, bias=True)

        if norm == 'batch':
            self.norm1 = nn.BatchNorm1d(hidden_dim_one, affine=norm_affine)
            self.norm2 = nn.BatchNorm1d(hidden_dim_two, affine=norm_affine)
        if norm == 'layer':
            self.norm1 = nn.LayerNorm(hidden_dim_one, elementwise_affine = norm_affine)
            self.norm2 = nn.LayerNorm(hidden_dim_two, elementwise_affine = norm_affine)
        if norm is None:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        self.activation = F.relu
        self.output_dim = output_dim
        self.num_seen = 0

        if class_names is None:
            class_names = tuple([str(c) for c in range(output_dim)])
        self.class_names = class_names

        #self.best_weights = None
        #self.rewind_weights = None
        self.initialize(seed)
        #self.initial_weights = to_vector(self.parameters()).clone().detach()
        #self.initial_gradients = None

        self.w_history = [to_vector(self.parameters()).clone().detach().cpu()]
        self.g_history = []

    def initialize(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        #self.best_weights = None
        #self.rewind_weights = None
        #self.initial_weights = to_vector(self.parameters()).clone().detach()
        #self.initial_gradients = None
        #print('initial weights', self.initial_weights.shape)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        x = self.norm1(x)
        x = self.activation(self.fc2(x))
        x = self.norm2(x)
        x = F.log_softmax(self.fc3(x), dim=-1)
        return x

    def score(self, pred_labels, true_labels, metrics='acc'):
        score = None
        pred_labels = pred_labels.detach().cpu().numpy()
        true_labels = true_labels.detach().cpu().numpy()
        if metrics == 'acc':
            score = accuracy_score(true_labels, pred_labels, normalize=True)
            #assert pred_labels.size() == true_labels.size()
            #score = (pred_labels == true_labels)
            #return torch.mean(score.to(dtype=torch.float32))
        elif metrics == 'f1-score':
            score = f1_score(true_labels, pred_labels, average='weighted')
        elif metrics == 'precision':
            score = precision_score(true_labels, pred_labels, average='weighted')
        elif metrics == 'recall':
            score = recall_score(true_labels, pred_labels, average='weighted')
        elif metrics == 'confusion_matrix':
            cm = confusion_matrix(true_labels, pred_labels)
            return cm
        return score # return numpy float, not torch.Tensor([score])

    def feedback(self, log_probs, true_labels, reward='nce', metrics='acc', return_probs=False, reduce='mean'):
        """Compute batch score"""
        feedback = None
        #log_probs.cpu()
        #true_labels.cpu()
        if reward == metrics:
            pred_labels = torch.flatten(torch.argmax(log_probs, dim=1))
            return self.score(pred_labels, true_labels, metrics=metrics) # numpy float
        elif reward == 'nce':
            # negative cross entropy (supervised)
            # same as -F.nll_loss(log_probs, true_labels)
            one_hot = F.one_hot(true_labels, num_classes=self.output_dim).to(dtype=log_probs.dtype)
            feedback = (one_hot * log_probs).mean(dim=1)
            # also same as
            # feedback = torch.flatten(torch.gather(log_probs, dim=1, index=true_labels.view(-1,1)))

            # check negative log likelihood values (against pytorch's F.nll_loss)
            #feedback_ref = -F.nll_loss(log_probs, targets, reduction='mean')
            #assert torch.allclose(feedback, feedback_ref), (feedback, feedback_ref)
        elif reward == 'expected_reward':
            # expected (supervised)
            probs = torch.exp(log_probs)
            feedback = torch.flatten(torch.gather(probs, dim=1, index=true_labels.view(-1,1)))
        elif reward == 'sampled_score':
            # same as -F.nll_loss(log_probs, sampled_labels)
            probs = torch.exp(log_probs)
            sampled_labels = torch.flatten(torch.multinomial(probs, num_samples=1))
            feedback = self.score(sampled_labels, true_labels, metrics=metrics)
            probs = torch.flatten(torch.gather(log_probs, dim=1, index=sampled_labels.view(-1,1)))
            if return_probs:
                if reduce == 'mean':
                    return feedback, torch.mean(probs.to(dtype=torch.float32))
                elif reduce == 'sum':
                    return feedback, torch.sum(feedback.to(dtype=torch.float32))
        if reduce == 'mean':
            return torch.mean(feedback.to(dtype=torch.float32))
        elif reduce == 'sum':
            return torch.sum(feedback.to(dtype=torch.float32))

    def __repr__(self):
        return "<FullyConnectedNN()>"


class CNNModule(torch.nn.Module):
    def __init__(self, convolutions, previous_channels, norm=None, norm_affine = False):
        super(CNNModule, self).__init__()
        self.conv1 = nn.Conv2d(previous_channels, convolutions, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(convolutions, convolutions, 3, stride=1, padding=1)
        self.mp = nn.MaxPool2d(2, 2)
        self.norm = norm
        if self.norm == 'batch':
            self.norm1 = torch.nn.BatchNorm2d(convolutions, affine=norm_affine)
            self.norm2 = torch.nn.BatchNorm2d(convolutions, affine=norm_affine) 
        else:
            self.norm1 = torch.nn.Identity()
            self.norm2 = torch.nn.Identity()


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.norm1(x)
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.mp(x)
        return x


class ConvolutionalNN(FullyConnectedNN):
    def __init__(self, modules=1, fc1=256, fc2=256, output_dim=10, initial_module_size=64, input_size=32, seed=42,
                 class_names=None, norm=None, norm_affine=False):
        super(FullyConnectedNN, self).__init__()
        
        if class_names is None:
            class_names = tuple([str(c) for c in range(output_dim)])
        self.class_names = class_names
        conv_list = [CNNModule(initial_module_size, 3, norm=True)]
        prev_module_size = initial_module_size #64
        next_module_size = prev_module_size * 2
        for i in range(modules - 1):
            conv_list.append(CNNModule(next_module_size, prev_module_size, norm=True))
            prev_module_size = next_module_size
            next_module_size = prev_module_size * 2

        self.convs = torch.nn.ModuleList(conv_list)
        output_size = input_size
        for i in range(modules):
            ## Hardcoded values for calculating the change in size from convolutions.
            output_size = (((output_size - 3 + 2 * 1) / 1) + 1) / 2
        fc_input = prev_module_size * output_size ** 2
        fc_input = int(fc_input) 
        self.fc1 = torch.nn.Linear(fc_input, fc1)
        self.fc2 = torch.nn.Linear(fc1, fc2)
        self.fc3 = torch.nn.Linear(fc2, output_dim)

        if norm == 'batch':
            self.norm1 = torch.nn.BatchNorm1d(fc1, affine=norm_affine)
            self.norm2 = torch.nn.BatchNorm1d(fc2, affine=norm_affine)
        elif norm == 'layer':
            self.norm1 = torch.nn.LayerNorm(fc1, elementwise_affine=norm_affine)
            self.norm2 = torch.nn.LayerNorm(fc2, elementwise_affine=norm_affine)
        else:
            self.norm1 = torch.nn.Identity()
            self.norm2 = torch.nn.Identity()
            
        self.output_dim = output_dim
        self.num_seen = 0

        if class_names is None:
            class_names = tuple([str(c) for c in range(output_dim)])
        self.class_names = class_names

        #self.best_weights = None
        #self.rewind_weights = None
        self.initialize(seed)
        #self.initial_weights = to_vector(self.parameters()).clone().detach()
        #self.initial_gradients = None

        self.w_history = [to_vector(self.parameters()).clone().detach().cpu()]
        self.g_history = []

    def forward(self, x):
        for cnn in self.convs:
            x = cnn(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.norm1(x)
        x = F.relu(self.fc2(x))
        x = self.norm2(x)
        x = F.log_softmax(self.fc3(x))
        return x

    def initialize(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)

        for conv in self.convs:
            torch.nn.init.xavier_normal_(conv.conv1.weight)
            torch.nn.init.xavier_normal_(conv.conv2.weight)


    def __repr__(self):
        return "<ConvolutionalNN()>"

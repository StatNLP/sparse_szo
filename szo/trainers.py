# -*- coding: utf-8 -*-

"""
Trainers module: defines training procedure
"""

import numpy as np
import math
import scipy

import torch
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from util import pearsonr, corrcoef

from time import time
from collections import defaultdict

import tqdm
import logging
logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, model, optimizer, scheduler, num_epochs, num_rounds, label,
                 seed=123, init='reset', pruning_rate=0.2, reward='argmax', metrics=['acc'],
                 log_dir='runs', eval_interval=1000, masking_strategy='L1', device='cpu'):

        self.model = model
        self.optimizer = optimizer
        self.reward = reward
        self.metrics = metrics
        self.obj_metrics = metrics[0] # metrics to be used
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.num_rounds = num_rounds
        self.seed = seed
        self.init = init
        self.pruning_rate = pruning_rate
        self.masking_strategy = masking_strategy
        #self.rewind_step = rewind_step
        self._interval_offset = eval_interval
        self._interval = self._interval_offset
        self.device = device

        # for logging
        self.label = label
        self.log_dir = log_dir
        self.tb_writer = SummaryWriter(self.log_dir+'/'+self.label)
        self.steps = 0
        self.history = []

        self._w_prev = None
        self._g_prev = None

        self.return_probs = True if 'Bandit' in self.optimizer.__repr__() else False
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

        self.duration = 0.0
        self.threshold = 1.0e-6

    def validate(self, dataloader, mode='test'):
        logger.debug(f'Validation - mode: {mode}')
        results = {}
        running_mean = defaultdict(float)

        # eval data
        for batch_idx, (images, labels) in enumerate(dataloader, start=1):
            eval_images = images.to(self.device)
            eval_labels = labels.to(self.device)

            # whole batch evaluation on test/dev set
            with torch.no_grad():
                log_probs = self.model(eval_images)
                pred_labels = torch.argmax(log_probs, dim=1)
                results['reward'] = self.model.feedback(log_probs, eval_labels, reward=self.reward,
                                                       metrics=self.obj_metrics, return_probs=False)
                #assert pred_labels.size() == eval_images.size()

            # score # numpy float
            for metric in self.metrics:
                score = self.model.score(torch.flatten(pred_labels), torch.flatten(eval_labels), metrics=metric)
                running_mean[metric] += (score - running_mean[metric]) / batch_idx
            # confusion matrix
            #results['cm'] = self.model.score(torch.flatten(pred_labels), torch.flatten(eval_labels),
            #                                 metrics='confusion_matrix')

        results.update(running_mean)
        logger.debug(f'Validation - mode: {mode} - results: {results}')

        # estimate lipschitz_neighbour on the whole test set
        if mode == 'test':

            def accumulate_gradients_on_test(): # need gradientns
                #cuml_gradients = torch.zeros_like(self._w_dim)
                example_count = 0
                #self.optimizer.zero_grad()

                for images, labels in dataloader:
                    eval_images = images.to(self.device)
                    example_count += images.shape[0]

                    if self.return_probs:
                        _, rewards = self.model.feedback(self.model(eval_images), labels, reward=self.reward,
                                        metrics=self.obj_metrics, return_probs=self.return_probs, reduce="sum")
                    else:
                        rewards = self.model.feedback(self.model(eval_images), labels, reward=self.reward,
                                        metrics=self.obj_metrics, return_probs=self.return_probs, reduce="sum")

                    if self.n_gpu > 1:
                        rewards = rewards.sum()

                    rewards.backward() # compute gradients
                    #cuml_gradients += self.optimizer._gradients_to_vector()
                    #self.optimizer.zero_grad()

                #cuml_gradients /= example_count
                #return example_count
                return example_count

            results['test_grad_norm'], results['lipschitz_neighbour'] = self.optimizer.estimate(accumulate_gradients_on_test)
            logger.debug(f'Validation - mode: {mode} results: {results}')

        # write in tensorboard
        self.write_results(results, mode)

        return results


    def run_batch(self, trainloader, testloader, devloader, train_running_mean, offset, desc=''):
        """ perform one epoch loop

            for batch in batches:
                1. update network
                2. evaluate outputs
                3. log results
        """
        start = time()
        reward = 0.0

        tqdm_batch_iteration = tqdm.tqdm(enumerate(trainloader, start=offset), desc='Training')
        results = ''
        self.model.train()
        for batch_idx, (images, labels) in tqdm_batch_iteration:
            # global steps (discontinuous)
            self.steps += images.shape[0]

            tqdm_batch_iteration.set_description(f"{desc} -- Updates:{batch_idx} -- Examples:{self.steps}{results} -- Training -- ", refresh=True)

            # training data
            images = images.to(self.device)
            labels = labels.to(self.device)
            def closure():
                return self.model.feedback(self.model(images), labels, reward=self.reward,
                                           metrics=self.obj_metrics, return_probs=self.return_probs)

            # keep log before update (for lipschitz)
            if self.steps >= self._interval:
                self._w_prev = self.optimizer.w.clone().detach().cpu()
                true_grad = self.optimizer.get_first_order_gradients(closure, return_probs=self.return_probs)
                self._g_prev = true_grad.detach().cpu()

            # step
            start_update = time()
            self.optimizer.step(closure)
            end_update = time()
            self.duration += end_update - start_update
            assert batch_idx == self.optimizer.update_counter, (batch_idx, self.optimizer.update_counter)

            # store initial gradients after first update
            if len(self.model.g_history) < 1:
                true_grad = self.optimizer.get_first_order_gradients(closure, return_probs=self.return_probs)
                self.model.g_history.append(true_grad.detach().cpu())
                assert len(self.model.w_history) == len(self.model.g_history)

            # store rewind weights
            #if (self.rewind_step is None) and (self.rewind_step > self.steps):
            #    self.model.rewind_weights = self.optimizer.w

            # evaluate on train set
            self.model.eval()
            with torch.no_grad():
                log_probs = self.model(images)
                pred_labels = torch.argmax(log_probs, dim=1)
                reward = self.model.feedback(log_probs, labels, reward=self.reward,
                                             metrics=self.obj_metrics, return_probs=False)

            train_running_mean['reward'] += (reward - train_running_mean['reward']) / batch_idx
            for m in self.metrics:
                score = self.model.score(torch.flatten(pred_labels), torch.flatten(labels), metrics=m) # numpy float
                train_running_mean[m] += (score - train_running_mean[m]) / batch_idx

            # log
            if self.steps >= self._interval:
                tqdm_batch_iteration.set_description(f"{desc} -- Updates:{batch_idx} -- Examples:{self.steps}{results} -- Evaluating --", refresh=True)

                logger.debug(f'Evaluating at {self._interval}l ...')
                #_copy_w = self.optimizer.w
                #_copy_g = self.optimizer.g

                if hasattr(self.optimizer, 'coeff'):
                    train_running_mean['train_approx_grad_var'], train_running_mean['train_func_value_var'] = self.optimizer.get_approx_grad_var(closure, return_probs=self.return_probs)
                self.write_results(train_running_mean, 'train')

                #log history
                self.model.w_history.append(self.optimizer.w.cpu())
                true_grad = self.optimizer.get_first_order_gradients(closure, return_probs=self.return_probs)
                self.model.g_history.append(true_grad.detach().cpu())

                # evaluate on test and dev set
                test_results = self.validate(testloader, mode='test')
                dev_results = self.validate(devloader, mode='dev')

                # check the best results
                #self.check_best_results(train_running_mean, 'train')
                #self.check_best_results(test_metrics, 'test')
                #self.check_best_results(dev_metrics, 'dev')

                # increment offset
                self._interval += self._interval_offset

                #assert torch.allclose(_copy_w, self.optimizer.w)
                #assert torch.allclose(_copy_g, self.optimizer.g)

                results = f" -- train loss: {-train_running_mean['reward']} -- test acc: {test_results[self.obj_metrics]}"

            self.model.train()

        return train_running_mean, test_results, dev_results, (time()-start)/60, batch_idx

    def write_results(self, results, mode='train'):
        """write results in SummaryWriter"""
        prefix = ""
        if mode == 'train':
            prefix = "cuml_"

        self.tb_writer.add_scalar(f"{mode}/{prefix}reward_{self.reward}", results['reward'], self.steps)

        if mode in ['test', 'dev']:
            # evaluation metrics
            for m in self.metrics:
                self.tb_writer.add_scalar(f"{mode}/{prefix}{m}", results[m], self.steps)

            # confusion matrix (class-wise performance)
            #fig = plot_confusion_matrix(results['cm‘], self.model.class_names, output_path=None)
            #self.tb_writer.add_figure(f"{mode}/confusion_matrix.", fig, self.steps)

        if mode == 'test':
            for e, lipschitz_n in results['lipschitz_neighbour'].items():
                self.tb_writer.add_histogram(f"lipschitz/neighbourhood_{e}", torch.tensor(lipschitz_n), self.steps)
                self.tb_writer.add_scalar(f"lipschitz/neighbourhood_{e}_test", max(lipschitz_n), self.steps)
            self.tb_writer.add_scalar(f"gradients/norm_true_test", results['test_grad_norm'], self.steps)

        if mode == 'train':
            # evaluation metrics
            for m in self.metrics:
                self.tb_writer.add_scalar(f"{mode}/{prefix}{m}", results[m], self.steps)

            # plot approx gradients statistics
            g_approx = self.optimizer.g.detach().cpu()
            g_approx_norm = torch.norm(g_approx, p=2) ** 2 # = || g_\mu(w^t) ||^2
            self.tb_writer.add_scalar("gradients/norm_approx_train", g_approx_norm, self.steps)
            g_approx_abs = torch.abs(g_approx)
            self.tb_writer.add_histogram("gradients/approx_train", g_approx_abs, self.steps)

            w_abs = torch.abs(self.optimizer.w.cpu())

            # plot true gradients statistics
            if len(self.model.g_history) > 1:
                g_true = self.model.g_history[-1]
                g_true_norm = torch.norm(g_true, p=2) ** 2 # = || \nabla f(w^t) ||^2
                self.tb_writer.add_scalar("gradients/norm_true_train", g_true_norm, self.steps)
                g_true_abs = torch.abs(g_true)
                self.tb_writer.add_histogram("gradients/true_train", g_true_abs, self.steps)
                #zero_count = torch.sum(g_true == 0.0)
                #self.tb_writer.add_scalar("sparsity/true_train", zero_count/g_dim, self.steps)


                # sparsity
                g_dim = list(g_true.shape)[0]
                #g_true_over_zero = torch.where(g_true_abs > self.threshold, g_true, torch.zeros(g_dim).cpu())
                #g_true_near_zero_count = torch.sum(g_true_abs < self.threshold)
                #g_true_over_zero_count = torch.sum(g_true_abs >= self.threshold)
                #g_true_nonzero_count = torch.sum(g_true != 0.0)
                #g_true_zero_count = torch.sum(g_true == 0.0)
                #g_approx_over_zero = torch.where(g_approx_abs > self.threshold, g_approx, torch.zeros(g_dim).cpu())
                #g_approx_near_zero_count = torch.sum(g_approx_abs < self.threshold)
                #g_approx_over_zero_count = torch.sum(g_approx_abs >= self.threshold)
                #g_approx_nonzero_count = torch.sum(g_approx != 0.0)
                #g_approx_zero_count = torch.sum(g_approx == 0.0)
                #assert g_approx_near_zero_count + g_approx_over_zero_count == g_dim, (g_approx_near_zero_count, g_approx_over_zero_count, g_dim)
                #assert g_true_near_zero_count + g_true_over_zero_count == g_dim, (g_true_near_zero_count, g_true_over_zero_count, g_dim)


                if self.optimizer.prune_or_freeze == 'none':
                    # save true grads
                    np.save(f'{self.log_dir}/g_true_{self.steps}.npy', g_true.numpy())
                    ground_truth = g_true
                else:
                    ground_truth_log_dir = self.log_dir.replace('freeze', 'none').replace('prune', 'none').replace('heldout', 'none').replace('random', 'none').replace('L1', 'none')
                    ground_truth = torch.tensor(np.load(f'{ground_truth_log_dir}/g_true_{self. steps}.npy')).cpu()
                    if ground_truth is None:
                        raise Error

                ground_truth_abs = torch.abs(ground_truth)
                #tp_filter = torch.zeros(g_dim).cpu()
                tn_filter = torch.zeros(g_dim).cpu()
                fp_filter = torch.zeros(g_dim).cpu()
                fn_filter = torch.zeros(g_dim).cpu()
                tp_count = 0.0
                g_true_exact_zero_count = 0.0
                g_true_near_zero_count = 0.0
                g_true_over_zero_count = 0.0
                g_approx_exact_zero_count = 0.0
                g_approx_near_zero_count = 0.0
                g_approx_over_zero_count = 0.0
                for i in range(g_dim):
                    #tp_filter[i] = 1.0 if (ground_truth[i].item() < self.threshold and g_approx[i].item() < self.threshold) else 0.0
                    tn_filter[i] = 1.0 if (ground_truth[i].item() >= self.threshold and g_approx[i].item() >= self.threshold) else 0.0
                    fp_filter[i] = 1.0 if (ground_truth[i].item() >= self.threshold and g_approx[i].item() < self.threshold) else 0.0
                    fn_filter[i] = 1.0 if (ground_truth[i].item() < self.threshold and g_approx[i].item() >= self.threshold) else 0.0
                    tp_count += 1.0 if (ground_truth[i].item() < self.threshold and g_approx[i].item() < self.threshold) else 0.0
                    g_true_exact_zero_count += 1.0 if ground_truth[i].item() == 0.0 else 0.0
                    g_true_near_zero_count += 1.0 if ground_truth[i].item() < self.threshold else 0.0
                    g_true_over_zero_count += 1.0 if ground_truth[i].item() >= self.threshold else 0.0
                    g_approx_exact_zero_count += 1.0 if g_approx[i].item() == 0.0 else 0.0
                    g_approx_near_zero_count += 1.0 if g_approx[i].item() < self.threshold else 0.0
                    g_approx_over_zero_count += 1.0 if g_approx[i].item() >= self.threshold else 0.0

                #true_positives = torch.where(tp_filter.to(dtype=torch.bool), ground_truth_abs, torch.zeros(g_dim).cpu())
                true_negatives = torch.where(tn_filter.to(dtype=torch.bool), torch.abs(ground_truth - g_approx), torch.zeros(g_dim).cpu())
                true_negatives_abs = torch.where(tn_filter.to(dtype=torch.bool), torch.abs(ground_truth_abs - g_approx_abs), torch.zeros(g_dim).cpu())
                false_positives = torch.where(fp_filter.to(dtype=torch.bool), ground_truth_abs, torch.zeros(g_dim).cpu())
                false_negatives = torch.where(fn_filter.to(dtype=torch.bool), g_approx_abs, torch.zeros(g_dim).cpu())
                assert g_approx_near_zero_count + g_approx_over_zero_count == g_dim, (g_approx_near_zero_count, g_approx_over_zero_count, g_dim)
                assert g_true_near_zero_count + g_true_over_zero_count == g_dim, (g_true_near_zero_count, g_true_over_zero_count, g_dim)
                assert tp_count <= g_true_near_zero_count, (tp_count, g_true_near_zero_count)
                assert tp_count <= g_approx_near_zero_count, (tp_count, g_approx_near_zero_count)
                assert torch.sum(tn_filter) <= g_true_over_zero_count, (torch.sum(tn_filter), g_true_over_zero_count)
                assert torch.sum(tn_filter) <= g_approx_over_zero_count, (torch.sum(tn_filter), g_approx_over_zero_count)
                self.tb_writer.add_scalar("sparsity/g_approx_zero_count", g_approx_exact_zero_count, self.steps)
                self.tb_writer.add_scalar("sparsity/g_approx_near_zero_count", g_approx_near_zero_count, self.steps)
                self.tb_writer.add_scalar("sparsity/g_true_zero_count", g_true_exact_zero_count, self.steps)
                self.tb_writer.add_scalar("sparsity/g_true_near_zero_count", g_true_near_zero_count, self.steps)
                self.tb_writer.add_scalar("sparsity/true_positives_count", tp_count, self.steps)
                self.tb_writer.add_scalar("sparsity/true_negatives_count", torch.sum(tn_filter), self.steps)
                self.tb_writer.add_scalar("sparsity/false_positives_count", torch.sum(fp_filter), self.steps)
                self.tb_writer.add_scalar("sparsity/false_negatives_count", torch.sum(fn_filter), self.steps)
                self.tb_writer.add_scalar("sparsity/relative_error_true_negatives", torch.sum(true_negatives), self.steps)
                self.tb_writer.add_scalar("sparsity/relative_error_true_negatives_abs", torch.sum(true_negatives_abs), self.steps)
                self.tb_writer.add_scalar("sparsity/relative_error_false_negatives", torch.sum(false_negatives), self.steps)
                self.tb_writer.add_scalar("sparsity/relative_error_false_positives", torch.sum(false_positives), self.steps)
                self.tb_writer.add_scalar("sparsity/correl_true_approx", pearsonr(ground_truth_abs, g_approx_abs), self.steps)
                self.tb_writer.add_scalar("sparsity/correl_true_weights", pearsonr(ground_truth_abs, w_abs), self.steps)
                self.tb_writer.add_scalar("sparsity/correl_approx_weights", pearsonr(g_approx_abs, w_abs), self.steps)
                del ground_truth
                del ground_truth_abs

            # plot weights
            self.tb_writer.add_histogram("weights", w_abs, self.steps)
            #for name, params in self.model.named_parameters():
            #    if name.endswith('weight'):
            #        self.tb_writer.add_histogram("magnitude/"+name, torch.abs(params), self.steps)
            #w_var, w_mean = torch.var_mean(w_abs)
            #self.tb_writer.add_scalar("util/weight_var", torch.var(self.optimizer.g), self.steps)
            #self.tb_writer.add_scalar("util/weight_mean", torch.mean(self.optimizer.g), self.steps)

            # plot lipschitz
            lipschitz, lipschitz_numerator, lipschitz_denominator = 0.0, 0.0, 0.0
            assert len(self.model.w_history) == len(self.model.g_history)

            #### 1. local lipschitz_t-1_t
            if self._w_prev is not None and self._g_prev is not None:
                L_t_numerator = torch.norm(self._w_prev - self.model.w_history[-1], p=2)
                L_t_denominator = torch.norm(self._g_prev - self.model.g_history[-1], p=2)
                L_t = L_t_numerator / L_t_denominator
                self.tb_writer.add_scalar("lipschitz/t-1_t", L_t, self.steps)
                self.tb_writer.add_scalar("lipschitz/t-1_t_numerator", L_t_numerator, self.steps)
                self.tb_writer.add_scalar("lipschitz/t-1_t_denominator", L_t_denominator, self.steps)
                self._w_prev = None
                self._g_prev = None

            #### 2. global lipschitz_i_j
            if len(self.model.w_history) > 2:
                sample_size = min(len(self.model.w_history), 100)
                logger.debug(f'global lipschitz sample_size {sample_size}')
                index_pairs = []
                while len(index_pairs) < sample_size:
                    p = np.random.randint(0, high=sample_size, size=2)
                    i, j = p[0], p[1]
                    if i == j:
                        continue
                    elif i > j:
                        tmp = i
                        i = j
                        j = tmp
                    if (i, j) in index_pairs:
                        continue
                    index_pairs.append((i, j))
                    L_numerator = torch.norm(self.model.w_history[i] - self.model.w_history[j], p=2)
                    L_denominator = torch.norm(self.model.g_history[i] - self.model.g_history[j], p=2)
                    assert L_denominator.item() != 0.0
                    L =  L_numerator / L_denominator
                    if L > lipschitz:
                        lipschitz = L
                        lipschitz_numerator = L_numerator
                        lipschitz_denominator = L_denominator
                self.tb_writer.add_scalar("lipschitz/global", lipschitz, self.steps)
                self.tb_writer.add_scalar("lipschitz/global_numerator", lipschitz_numerator, self.steps)
                self.tb_writer.add_scalar("lipschitz/global_denominator", lipschitz_denominator, self.steps)

            #### 3. lipschitz_f-mu_f
            # || \nabla f(\w + \mu\u) - \nabla f(\w) || / || \mu\u ||
            if hasattr(self.optimizer, 'perturbations'):
                lipschitz_numerator = torch.norm(g_approx - self.model.g_history[-1], p=2)
                lipschitz_denominator = torch.norm(torch.mean(self.optimizer.perturbations, dim=0), p=2)
                self.tb_writer.add_scalar("lipschitz/f-mu_f", lipschitz_numerator / lipschitz_denominator, self.steps)
                self.tb_writer.add_scalar("lipschitz/f-mu_f_numerator", lipschitz_numerator, self.steps)
                self.tb_writer.add_scalar("lipschitz/f-mu_f_denominator", lipschitz_denominator, self.steps)

            if 'train_approx_grad_var' in results.keys():
                self.tb_writer.add_scalar("gradients/f-mu_f_var", results['train_approx_grad_var'], self.steps)
            if 'train_func_value_var' in results.keys():
                self.tb_writer.add_scalar("gradients/coeff_var", results['train_func_value_var'], self.steps)

            # learning rate
            #self.tb_writer.add_scalar("util/learning_rate", self.optimizer.lr, self.steps)

        return 0

    #def check_best_results(self, metrics, mode='train'):
    #    for m, v in metrics.items():
    #        if v > self.history[-1]['best_'+mode+'_'+m]:
    #            self.history[-1]['best_'+mode+'_'+m] = v
    #            self.history[-1]['best_'+mode+'_'+m+'_step'] = (self.optimizer.update_counter, self.steps)
    #            #if mode == 'test' and m == self.obj_metrics:
    #            #    self.model.best_weights = self.optimizer.w

    def prune(self, devloader, r):
        """ Prune network """
        sparsity_before, num_active_before = self.optimizer.check_sparsity()
        assert sparsity_before == self.history[r]['sparsity']

        # survive vec : values on mask = 1
        if self.init == 'random':
            # next start weights <= random values
            self.model.initialize(seed=self.seed+r+1)
            survive_vec = self.model.initial_weights
        elif self.init == 'reset':
            # next start weights <= orig weight values at iteration 0
            survive_vec = self.model.initial_weights
        #elif init == 'best':
        # next start weights <= weight values at the best dev acc so far
        #survive_vec = self.model.best_weights
        #elif init == 'rewind':
        # next start weights <= weight values at iter t
        #survive_vec = self.model.rewind_weights
        elif self.init == 'last':
            # next start weights <= prev last weights
            survive_vec = self.optimizer.w
        else:
            raise ValueError(f'initialization method {self.init} not found.')

        num_to_prune = np.floor(self.pruning_rate * num_active_before).astype(np.int32)
        offset_to_prune = sum([h['num_to_prune'] for h in self.history])

        # call self.optimizer.prune()
        if self.masking_strategy == 'L1':
            magnitude = self.optimizer.prune(survive_vec, offset_to_prune + num_to_prune, None)
        elif self.masking_strategy == 'heldout':
            self.model.eval()

            # return numpy float
            def dev_score_closure():
                dev_running_mean = 0.0

                # eval on whole dev data
                for batch_idx, (images, labels) in enumerate(devloader, start=1):
                    dev_images = images.to(self.device)
                    dev_labels = labels.to(self.device)

                    with torch.no_grad():
                        pred_labels = torch.argmax(self.model(dev_images), dim=1)
                        score = self.model.score(torch.flatten(pred_labels), torch.flatten(dev_labels),
                                                metrics=self.obj_metrics) # numpy float
                    dev_running_mean += (score - dev_running_mean) / batch_idx

                return dev_running_mean # return numpy float (not torch.tensor)

            _ = self.optimizer.prune(survive_vec, offset_to_prune + num_to_prune, dev_score_closure)

        elif self.masking_strategy == 'random':
            _ = self.optimizer.prune(survive_vec, offset_to_prune + num_to_prune, 'random')

        # log
        sparsity_after, num_active_after = self.optimizer.check_sparsity()
        logger.info(f'Prune (total dim: {self.optimizer.w.numel()}):')
        logger.info(f' - {self.pruning_rate*100}% ({num_to_prune}) dim masked.')
        if self.masking_strategy == 'L1':
            logger.info(f' - cutoff magnitude: {magnitude}')
        log_string = f' - before prune - sparsity: {sparsity_before} - num_active: {num_active_before}'
        if self.masking_strategy == 'heldout' and self.optimizer.dev_scores is not None:
            log_string += f' - dev score: {self.optimizer.dev_scores[0]}'
        logger.info(log_string)
        log_string = f' -  after prune - sparsity: {sparsity_after} - num_active: {num_active_after}'
        if self.masking_strategy == 'heldout' and self.optimizer.dev_scores is not None:
            log_string += f' - best dev score: {self.optimizer.dev_scores[1]}'
            log_string += f' - worst dev score: {self.optimizer.dev_scores[2]}'
            log_string += f' - std: {self.optimizer.dev_scores[3]}'
        logger.info(log_string)

        self.optimizer.dev_scores = None

        return num_to_prune

    def initialize(self, r, num_to_prune):
        #assert r == len(self.history)
        # crete history dict
        history = {}
        #for m in self.metrics + ['reward']:
        #    for t in ['train', 'test', 'dev']:
        #        history['best_'+t+'_'+m] = 0.0
        #        history['best_'+t+'_'+m+'_step'] = 0

        # update pruning history
        sparsity, num_active = self.optimizer.check_sparsity()
        history['sparsity'] = sparsity
        history['num_active'] = num_active
        history['num_to_prune'] = num_to_prune

        # reset optimizer
        self.optimizer.initialize()

        # reset random seed every round
        torch.manual_seed(self.seed)

        # update history
        self.history.append(history)

    def train(self, trainloader, testloader, devloader):
        """ Train network

            for r in rounds:
                for e in epochs:
                    for b in batches:
                        update
                        evaluate
                prune
        """
        # reset place holders
        train_results = defaultdict(float) # #defaultdict(torch.tensor([]))
        offset = 0
        num_to_prune = 0
        np.random.seed(self.seed)

        # start training
        logger.info(f'***** Start Training ***** {self.optimizer.__repr__()}')
        start_train = time()
        for r in range(self.num_rounds):
            # clear variables
            self.initialize(r, num_to_prune)

            start_round = time()
            logger.info(f'Round # {r}')
            for e in range(self.num_epochs):
                logger.info(f'\tEpoch # {e}')

                # train
                train_results, test_results, dev_results, duration, offset = self.run_batch(
                    trainloader, testloader, devloader, train_results, offset+1, desc=f'Round:{r} -- Epoch:{e}')

                # log
                logger.info(f"\t\t - duration: {duration} [min]\t - learning rate: {self.optimizer.lr}")
                logger.info(f'\t\t'+''.join([f' - Train {m}: {train_results[m]}' for m in self.metrics + ['reward']]))
                logger.info(f'\t\t'+''.join([f' - Test {m}: {test_results[m]}' for m in self.metrics + ['reward']]))
                logger.info(f'\t\t'+''.join([f' - Dev {m}: {dev_results[m]}' for m in self.metrics + ['reward']]))

                # decay learning rate
                if self.scheduler:
                    self.scheduler.step(test_results[self.obj_metrics])

            logger.info(f"Update Time (cumulative) # {r}: {self.duration} [sec]")
            logger.info(f"Total Time in Round # {r}: {(time()-start_round)/60} [min]")

            # prune
            if self.optimizer.prune_or_freeze != 'none' and self.pruning_rate > 0.0 and r < self.num_rounds-1:
                num_to_prune = self.prune(devloader, r)

        logger.info(f"Update Time: {self.duration} [sec]")
        logger.info(f"Total Time: {(time()-start_train)/60} [min]")

        #self.history.append({'histogram': self._tmp_histogram})
        #ret = defaultdict(list)
        #for h in self.history:
        #    for k, v in h.items():
        #        ret[k].append(v)
        #torch.save(ret, 'logs/'+nself.log_dir+'-history')

        logger.info('***** End Training *****')

        #except Exception as e:
        #    logger.warning(e)
        #    sys.exit(1)


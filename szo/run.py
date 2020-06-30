# -*- coding: utf-8 -*-

"""
Main module: triggers training
"""


from torch.optim import lr_scheduler
import torch
from models import FullyConnectedNN, ConvolutionalNN
from optimizers import FirstOrderOptimizer, FirstOrderBanditOptimizer, VanillaEvolutionOptimizer, \
    DuelingEvolutionOptimizer, OneSideEvolutionOptimizer, TwoSideEvolutionOptimizer
from trainers import Trainer
from util import get_dataloader, mnist, cifar10

import os
import argparse

import logging
from util import TqdmLoggingHandler

def main():
    ap = argparse.ArgumentParser("SZO")
    ap.add_argument("--data", choices=["mnist", "cifar10"], default="mnist", help="dataset") #, "skewedmnist"
    ap.add_argument("--opt", choices=["first", "flaxman", "dueling", "ghadimi", "agarwal"], help="optimizer type")
    ap.add_argument("--model", choices=["fc3", "cnn"], help="Model type")
    ap.add_argument("--depth", default=1, type=int, help="Depth of the cnn")
    ap.add_argument("--seed", default=12345, type=int, help="random seed")
    ap.add_argument("--num_epochs", default=5, type=int, help="number of epochs")
    ap.add_argument("--num_rounds", default=20, type=int, help="number of rounds")
    ap.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    ap.add_argument("--pr", default=0.2, type=float, help="pruning rate")
    ap.add_argument("--mu", default=0.1, type=float, help="exploration rate, smoothing parameter")
    ap.add_argument("--beta", default=0.0, type=float, help="momentum")
    ap.add_argument("--max_grad_norm", default=0.0, type=float, help="maximum gradient norm")
    ap.add_argument("--var", default=1.0, type=float, help="noise variance")
    ap.add_argument("--eval_interval", default=10000, type=int, help="evaluation interval")
    ap.add_argument("--batch_size", default=64, type=int, help="batch_size")
    ap.add_argument("--eval_batch_size", default=1000, type=int, help="batch size used in evaluation")
    ap.add_argument("--cv", default=True, action="store_true", help="whether to include control variates") # type=bool,
    ap.add_argument("--init", choices=["reset", "random", "last"], #, 'rewind', 'best'
                    help="initialization strategy in pruning: one of {reset, random, last}") #, rewind, best
    #ap.add_argument("--rewind_step", type=int, help="which epoch to return to after pruning")
    ap.add_argument("--reward", choices=["nce", "acc", "expected_reward", "sampled_score"],
                    help="reward function: one of {nce, acc, expected_reward, sampled_score}")
    ap.add_argument("--prune_or_freeze", choices=["none", "prune", "freeze"],
                    help="sparsification strategy: one of {prune or freeze}")
    ap.add_argument("--masking_strategy", choices=["none", "L1", "heldout", "random"],
                    help="masking strategy: one of {none, L1, heldout, random}")
    ap.add_argument("--num_samples", type=int, help="number of samples to evaluate for gradient estimation")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument('--affine', action="store_true", default=False, # type=bool,
                    help="if specified, turn on affine transform in normalization layers")
    ap.add_argument('--norm', choices=["batch", "layer", "none"], default="batch",
                    help="type of normalization to use between NN layeres")

    args = ap.parse_args()


    log_dir = f'runs-{args.seed}'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    #if not os.path.exists('logs/'+log_dir):
    #    os.mkdir('logs/'+log_dir)

    # logging
    label = f'{args.opt}-{args.reward}-{args.prune_or_freeze}-{args.init}-{args.masking_strategy}-{args.batch_size}'
    logging.basicConfig(filename=os.path.join(log_dir, f'{label}-train.log'),
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.addHandler(TqdmLoggingHandler())

    logger.info('Arguments:')
    for arg in vars(args):
        logger.info(f'\t{arg}: {getattr(args, arg)}')

    # data
    if args.data == 'mnist':
        trainset, testset, classes = mnist(data_path='data/MNIST_data/')
    elif args.data == 'cifar10':
        trainset, testset, classes = cifar10(data_path='data/CIFAR10_data/')
    trainloader, testloader, devloader = get_dataloader(trainset, testset, batch_size=args.batch_size,
                                                        eval_batch_size=args.eval_batch_size, seed=args.seed)

    # model
    model = None
    model_kwargs = {'seed': args.seed, 'class_names': classes, 'output_dim': len(classes), 'norm_affine': args.affine, 'norm': args.norm}
    if args.model == 'cnn':
        assert args.data == 'cifar10'
        model_kwargs['modules'] = args.depth
        model_kwargs['input_size'] = 32
        model = ConvolutionalNN(**model_kwargs)
    elif args.model == 'fc3':
        if args.data == 'mnist':
            model_kwargs['input_dim'] = 28*28
        elif args.data == 'cifar10':
            model_kwargs['input_dim'] = 32*32*3
        model = FullyConnectedNN(**model_kwargs)
    else:
        raise ValueError("Unknown model type")

    # gpu
    device = None
    if args.device == 'gpu' and torch.cuda.is_available():
        device = 'cuda:0'
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        device = 'cpu'
    model.to(device)

    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"\tn_gpu: {torch.cuda.device_count()}")

    # optimizer
    kwargs = {'prune_or_freeze': args.prune_or_freeze, 'init': args.init}
    if args.lr:
        kwargs['lr'] = args.lr
    if args.mu:
        kwargs['mu'] = args.mu
    if args.beta:
        kwargs['beta'] = args.beta
    if args.max_grad_norm:
        kwargs['max_grad_norm'] = args.max_grad_norm
    if args.var:
        kwargs['var'] = args.var
    if args.num_samples:
        kwargs['num_samples'] = args.num_samples
    #if args.init == 'rewind':
    #    print(args.rewind_step)

    opt = None
    if args.opt == 'first':
        if args.reward in ['sampled_score']:
            kwargs['cv'] = args.cv # control variates
            opt = FirstOrderBanditOptimizer(model.parameters(), **kwargs)
        elif args.reward in ['nce', 'expected_reward']:
            opt = FirstOrderOptimizer(model.parameters(), **kwargs)
        else:
            raise ValueError
    elif args.opt == 'flaxman':
        opt = VanillaEvolutionOptimizer(model.parameters(), **kwargs)
    elif args.opt == 'dueling':
        opt = DuelingEvolutionOptimizer(model.parameters(), **kwargs)
    elif args.opt == 'ghadimi':
        opt = OneSideEvolutionOptimizer(model.parameters(), **kwargs)
    elif args.opt == 'agarwal':
        opt = TwoSideEvolutionOptimizer(model.parameters(), **kwargs)
    else:
        raise ValueError("Unknown optimizer type")


    #scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, threshold=1e-2)
    scheduler = None # constant learning rate
    # trainer
    pruning_rate = 0.0 if args.prune_or_freeze == 'none' or args.masking_strategy == 'none' else args.pr
    metrics = ['acc', 'f1-score', 'precision', 'recall']
    trainer = Trainer(model, opt, scheduler, args.num_epochs, args.num_rounds, label,
                      seed=args.seed, init=args.init, pruning_rate=pruning_rate, reward=args.reward,
                      metrics=metrics, log_dir=log_dir, eval_interval=args.eval_interval,
                      masking_strategy=args.masking_strategy, device=device)

    trainer.train(trainloader, testloader, devloader)

    #del model
    #del opt
    #del scheduler
    #del trainer

    logging.shutdown()

if __name__ == "__main__":
    main()

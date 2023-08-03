from ADDA import ADDA
from DANN import DANN
from IRM import IRM
from mlp import MLP
from ResNet import ResNet
from REx import REx
import pdb
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import argparse
from loguru import logger
import sys
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from Dataset import MyDataset, prepare_set, read_data
from sklearn.metrics import confusion_matrix
random.seed(114514)
np.random.seed(114514)
torch.manual_seed(114514)


def get_args():
    parser = argparse.ArgumentParser(description="General Traning Pipeline")

    parser.add_argument("--model", type=str, default='DANN')
    parser.add_argument("--svm_c", type=float, default=.1)
    parser.add_argument("--svm_kernel", choices=['linear', 'rbf', 'sigmoid', 'poly'], default='linear')
    parser.add_argument("--lamda", type=float, default=0.5)
    parser.add_argument("--triplet_weight", type=float, default=.1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--display_epoch", type=int, default=1)
    parser.add_argument("--penalty_weight", type=float, default=.5)
    parser.add_argument("--weight_decay", type=float, default=.0)
    parser.add_argument("--variance_weight", type=float, default=.5)
    parser.add_argument("--pretrain_epoch", type=int, default=500)
    parser.add_argument("--advtrain_iteration", type=int, default=10000)
    parser.add_argument("--critic_iters", type=int, default=10)
    parser.add_argument("--gen_iters", type=int, default=10000)
    parser.add_argument("--is_augmentation", action='store_true')
    parser.add_argument("--display_iters", type=int, default=10)

    return parser.parse_args()


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def get_logger(output_file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    # logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    while os.path.exists(output_file):
        output_file = output_file.replace('.log', '1.log')
    if output_file:
        logger.add(output_file, enqueue=True, format=log_format)
    return logger


args = get_args()
output = f'./loggers/train_{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.model}.log'
logger = get_logger(output)
logger.info(args)


def create_model(args):
    if args.model == 'DANN':
        return DANN(1024, args.hidden_dim, 4, 2, args.lamda)
    elif args.model == 'MLP':
        return MLP(1024, args.hidden_dim, 4)
    elif args.model == 'ResNet':
        return ResNet(1024, args.hidden_dim, 4)
    elif args.model == 'ADDA':
        return ADDA(1024, args.hidden_dim, 4, 1, 10)
    elif args.model == 'IRM':
        return IRM(1024, args.hidden_dim, 4, args.penalty_weight)
    elif args.model == 'REx':
        return REx(1024, args.hidden_dim, 4, args.variance_weight)
    else:
        raise ValueError("Unknown model type!")




def train_DANN(model, testers_data, testers_label):
    accuracy = 0.0
    model = model.to(device)
    train_data, train_label, test_data, test_label = prepare_set(testers_data, testers_label)

    train_set = MyDataset(train_data, train_label)
    test_set = MyDataset(test_data, test_label)

    test_data = torch.tensor(test_data)
    test_label = torch.tensor(test_label)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    test_iter = iter(test_loader)
    best_acc = 0.0
    for epoch in range(args.num_epoch):
        model.train()
        total_class_loss = 0.0
        total_domain_loss = 0.0
        for batch in train_loader:
            inputs, class_labels = batch
            domain_source_labels = torch.zeros(len(inputs)).long()
            data = inputs.to(device), class_labels.to(device), domain_source_labels.to(device)

            try:
                inputs, _ = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                inputs, _ = next(test_iter)

            domain_target_labels = torch.ones(len(inputs)).long()
            testData = inputs.to(device), domain_target_labels.to(device)

            class_loss, domain_loss = model.compute_loss(data, testData)
            loss = class_loss + domain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_class_loss += class_loss.detach().cpu().numpy()
            total_domain_loss += domain_loss.detach().cpu().numpy()
            # logger.info(
            #     f'epoch={epoch}, total class loss = {class_loss:.4f}, total domain loss = {domain_loss:.4f}')
        if epoch % args.display_epoch == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                all = 0
                pred_class_label, _ = model(test_data.to(device).float())
                pred = pred_class_label.argmax(dim=-1)
                correct += (pred == test_label.to(device)).sum().item()
                all += len(pred)
                acc = correct / all
                logger.info(f'epoch={epoch}, class_acc = {acc:.4f}')
                if acc > best_acc:
                    best_acc = acc
                pred = pred.detach().cpu().numpy()
                confusion = confusion_matrix(test_label, pred)
                for i in range(len(confusion)):
                    if np.sum(confusion[i, :]) == 0:
                        logger.info(f'There is no category{i} in test_set.')
                    else:
                        pre = round(confusion[i][i] / np.sum(confusion[i, :]), 4) * 100
                        logger.info(f'precision_{i}={pre}%')
    filename = f'train_{datetime.now():%Y-%m-%d_%H-%M-%S}' + f"DANN_checkpoint.pt"
    torch.save(model.state_dict(), 'model/' + filename)
    logger.info(f'accuracy = {best_acc:.4f}')


def train_ADDA(model, testers_data, testers_label):


    logger.info('---Pretraining---')
    train_data, train_label, test_data, test_label = prepare_set(testers_data, testers_label)

    train_set = MyDataset(train_data, train_label)
    test_set = MyDataset(test_data, test_label)

    test_data = torch.tensor(test_data)
    test_label = torch.tensor(test_label)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

    model.srcMapper.to(device)
    model.Classifier.to(device)
    model.tgtMapper.to(device)
    model.Discriminator.to(device)

    optimizer_src = torch.optim.Adam(model.srcMapper.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optimizer_cls = torch.optim.Adam(model.Classifier.parameters(), lr=args.lr, betas=(0.5, 0.9))

    best_acc = 0.0
    for epoch in range(args.pretrain_epoch):
        model.srcMapper.train()
        model.Classifier.train()
        total_loss = 0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device).float(), y.to(device).float()
            optimizer_src.zero_grad()
            optimizer_cls.zero_grad()
            loss = model.pretrain_loss((x, y))
            loss.requires_grad_(True)
            loss.backward()
            optimizer_src.step()
            optimizer_cls.step()
            total_loss += loss.detach().cpu().numpy()
            # logger.info(f'epoch={epoch}, pretrain loss = {loss:.4f}')
        if epoch % args.display_epoch == 0:
            model.srcMapper.eval()
            model.Classifier.eval()
            with torch.no_grad():
                correct = 0
                all = 0
                acc = 0.0
                num = 0
                result = model.Classifier(model.srcMapper(test_data.to(device).float()))
                pred = result.argmax(dim=-1)
                correct += (pred == test_label.to(device)).sum().item()
                all += len(pred)
                acc += correct / all
                num += 1
                logger.info(f'epoch={epoch}, class_acc = {correct / all:.4f}')
                acc /= num
                if acc > best_acc:
                    best_acc = acc
                pred = pred.detach().cpu().numpy()
                confusion = confusion_matrix(test_label, pred)
                for i in range(len(confusion)):
                    if np.sum(confusion[i, :]) == 0:
                        logger.info(f'There is no category{i} in test_set.')
                    else:
                        pre = round(confusion[i][i] / np.sum(confusion[i, :]), 4) * 100
                        logger.info(f'precision_{i}={pre}%')
    filename = f'train_{datetime.now():%Y-%m-%d_%H-%M-%S}' + "ADDN_Src_Mapper_checkpoint.pt"
    torch.save(model.srcMapper.state_dict(), 'model/' + filename)
    filename = f'train_{datetime.now():%Y-%m-%d_%H-%M-%S}' + "ADDN_Classifier_checkpoint.pt"
    torch.save(model.Classifier.state_dict(), 'model/' + filename)

    # adversarial train
    logger.info('-- Adversarial Training --')
    model.tgtMapper.load_state_dict(model.srcMapper.state_dict())
    optimizer_tgt = torch.optim.RMSprop(model.tgtMapper.parameters(), lr=args.lr)
    optimizer_disc = torch.optim.RMSprop(model.Discriminator.parameters(), lr=args.lr)

    train_iter = iter(train_loader)
    test_iter = iter(test_loader)

    for param in model.srcMapper.parameters():
        param.requires_grad = False
    for param in model.Classifier.parameters():
        param.requires_grad = False

    tgt_x = None
    best_acc = 0.
    for iteration in range(args.advtrain_iteration):
        model.tgtMapper.train()
        model.Discriminator.train()
        total_disc_loss = 0
        total_target_loss = 0

        for p in model.Discriminator.parameters():
            p.requires_grad = True

        for _ in range(args.critic_iters):
            try:
                src_x, _ = next(train_iter)
                if src_x.size(0) < args.batch_size:
                    train_iter = iter(train_loader)
                    src_x, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                src_x, _ = next(train_iter)
            src_x = src_x.to(device).float()

            try:
                tgt_x, _ = next(test_iter)
                if tgt_x.size(0) < args.batch_size:
                    test_iter = iter(test_loader)
                    tgt_x, _ = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                tgt_x, _ = next(test_iter)
            tgt_x = tgt_x.to(device).float()

            optimizer_disc.zero_grad()
            loss_discriminator = model.discriminator_loss(src_x, tgt_x)
            loss_discriminator.backward()
            optimizer_disc.step()

            total_disc_loss += loss_discriminator.detach().cpu().numpy() / args.critic_iters

        for p in model.Discriminator.parameters():
            p.requires_grad = False

        optimizer_disc.zero_grad()
        optimizer_tgt.zero_grad()
        loss_tgt = model.tgt_loss(tgt_x)
        loss_tgt.backward()
        optimizer_tgt.step()

        total_target_loss += loss_tgt.detach().cpu().numpy()
        model.tgtMapper.eval()
        model.Discriminator.eval()
        with torch.no_grad():
            correct = 0
            all = 0
            acc = 0.0
            num = 0
            result = model.Classifier(model.tgtMapper(test_data.to(device).float()))
            pred = result.argmax(dim=-1)
            correct += (pred == test_label.to(device)).sum().item()
            all += len(pred)
            acc += correct / all
            num += 1
            logger.info(f'iteration={iteration}, test_acc = {correct / all:.4f}')
            acc /= num
            if acc > best_acc:
                best_acc = acc
            logger.info(
                f"Iteration {iteration}, Discriminator Loss {total_disc_loss:.4f}, Target Loss {total_target_loss:.4f}")
            pred = pred.detach().cpu().numpy()
            confusion = confusion_matrix(test_label, pred)
            for i in range(len(confusion)):
                if np.sum(confusion[i, :]) == 0:
                    logger.info(f'There is no category{i} in test_set.')
                else:
                    pre = round(confusion[i][i] / np.sum(confusion[i, :]), 4) * 100
                    logger.info(f'precision_{i}={pre}%')
    filename = f'train_{datetime.now():%Y-%m-%d_%H-%M-%S}' + "ADDA_tgt_Mapper_checkpoint.pt"
    torch.save(model.tgtMapper.state_dict(), 'model/' + filename)
    filename = f'train_{datetime.now():%Y-%m-%d_%H-%M-%S}' + "ADDA_Discriminator_checkpoint.pt"
    torch.save(model.Discriminator.state_dict(), 'model/' + filename)


def train_generalization(model, testers_data, testers_label):

    model = model.to(device)
    train_data, train_label, test_data, test_label = prepare_set(testers_data, testers_label)

    train_set = MyDataset(train_data, train_label)
    test_set = MyDataset(test_data, test_label)

    test_data = torch.tensor(test_data)
    test_label = torch.tensor(test_label)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_acc = 0.0
    for epoch in range(args.num_epoch):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device).float(), y.to(device).to(torch.int64)
            loss = model.compute_loss((x, y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().numpy()
        if epoch % args.display_epoch == 0:
            model.eval()
            acc = 0.0
            with torch.no_grad():
                result = model(test_data.to(device).float())
                pred = result.argmax(dim=-1)
                acc = (pred == test_label.to(device)).sum().item() / len(test_label)
            logger.info(f'epoch={epoch}, acc = {acc:.4f}')
            best_acc = max(acc, best_acc)
            pred = pred.detach().cpu().numpy()
            confusion = confusion_matrix(test_label, pred)
            for i in range(len(confusion)):
                if np.sum(confusion[i, :]) == 0:
                    logger.info(f'There is no category{i} in test_set.')
                else:
                    pre = round(confusion[i][i] / np.sum(confusion[i, :]), 4) * 100
                    logger.info(f'precision_{i}={pre}%')
    filename = f'train_{datetime.now():%Y-%m-%d_%H-%M-%S}' + f"{args.model}_checkpoint.pt"
    torch.save(model.state_dict(), 'model/' + filename)

model = create_model(args)
data, label = read_data()
if args.model == 'DANN':
    train_DANN(model, data, label)
elif args.model == 'ADDA':
    train_ADDA(model, data, label)
elif args.model == 'MLP' or args.model == 'ResNet' or args.model == 'IRM' or args.model == 'REx':
    train_generalization(model, data, label)
else:
    raise ValueError("Unknown model type!")

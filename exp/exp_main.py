
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import regre_metric,class_metric #metric
# from sklearn.metrics import corruracy_score,roc_mae_score, confusion_matrix
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import os
import time
import json
import pickle

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger('__main__')

from models import (
    BSGN,
    Ablation1,
    Ablation2,
    Remove1,
    Remove2,
    Remove3,
    Base1,
    Base2
)


class Exp_Main(Exp_Basic):
    def __init__(self, args,shuffle_ix, train_index, test_index):
        super(Exp_Main, self).__init__(args)
        self.shuffle_ix= shuffle_ix
        self.train_index = train_index
        self.test_index = test_index

    def _build_model(self):
        model_dict = {
            "BSGN":BSGN,
            'Ablation1':Ablation1,
            'Ablation2':Ablation2,
            'Remove1':Remove1,
            'Remove2':Remove2,
            'Remove3':Remove3,
            'Base1':Base1,
            'Base2':Base2,

        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args,self.shuffle_ix,self.train_index,self.test_index, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _select_metric(self,pred, true):
        pred = np.array(pred).squeeze()
        true = np.array(true).squeeze()
        metric1,metric2 = regre_metric(pred, true)
        return metric1,metric2


    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = 0.0
        true_list = []
        pred_list = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):

                pred, true = self._process_one_batch(batch_x, batch_y)

                loss = criterion(pred, true)
                total_loss += loss.item()
                true_list.extend(true.detach().cpu().numpy())
                pred_list.extend(pred.detach().cpu().numpy())

        total_loss = total_loss/len(vali_loader)

        metric1,metric2 = self._select_metric(pred_list,true_list)

        # regression
        logger.info("          Val   loss:{:.4f},    corr|||{:.4f}|||corr,   mae|||{:.4f}|||mae".format(total_loss,metric1,metric2))


        self.model.train()

        return total_loss

    def train(self, path):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            
            time_now = time.time()
            train_loss = 0.0
            true_list = []
            pred_list = []
        
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(batch_x, batch_y) # torch.Size([16, 1])

                loss = criterion(pred, true)
                loss.backward()
                model_optim.step()

                train_loss += loss.item()

                true_list.extend(true.detach().cpu().numpy())
                pred_list.extend(pred.detach().cpu().numpy())

            train_loss = train_loss/len(train_loader)

            logger.info("Epoch: {} cost time: {:.4f}".format(epoch + 1, time.time() - epoch_time))

            metric1,metric2 = self._select_metric(pred_list,true_list)
            logger.info("          train loss:{:.4f},    corr|||{:.4f}|||corr,   mae|||{:.4f}|||mae".format(train_loss,metric1,metric2))

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 早停时已保存最优模型，而当前不一定是最优模型，所以要先加载，再保存
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path + '/' + 'checkpoint.pth')

        return self.model

    def test(self, path, save_pred=False, inverse=False):
        
        test_data, test_loader = self._get_data(flag='test')
        
        # 加载最优模型
        logger.info('loading model')
        self.model.eval()
        self.model.load_state_dict(torch.load(path + '/' + 'checkpoint.pth'))

        true_list = []
        pred_list = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(batch_x, batch_y)
                
                true_list.extend(true.detach().cpu().numpy()) 
                pred_list.extend(pred.detach().cpu().numpy())

        metric1,metric2 = self._select_metric(pred_list,true_list)
        logger.info("                                corr|||{:.4f}|||corr,   mae|||{:.4f}|||mae".format(metric1,metric2))
        return metric1,metric2, pred_list, true_list
    

    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)


        return outputs, batch_y  # torch.Size([16, 1])


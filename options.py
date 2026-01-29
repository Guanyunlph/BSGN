import argparse
import torch


class Options(object):

    def __init__(self):
        
        self.parser = argparse.ArgumentParser(description='')

        self.parser.add_argument('--seed', type=int, default=0)
        self.parser.add_argument('--fold', type=int, default=10)
        # basic config       
        self.parser.add_argument('--model', type=str, default='BSGN')

        self.parser.add_argument('--network_id', type=int, default=999)

        # self.parser.add_argument('--data', type=str, default='camcan-movie', help='')
        # self.parser.add_argument('--d_subj_num', type=int, default=563, help='')
        # self.parser.add_argument('--seq_len', type=int, default=188, help='')  

        self.parser.add_argument('--data', type=str, default='camcan-rest', help='')
        self.parser.add_argument('--d_subj_num', type=int, default=595, help='')
        self.parser.add_argument('--seq_len', type=int, default=256, help='')  

        # self.parser.add_argument('--data', type=str, default='nki', help='')
        # self.parser.add_argument('--d_subj_num', type=int, default=1137, help='')
        # self.parser.add_argument('--seq_len', type=int, default=115, help='')  

        # model define
        self.parser.add_argument('--d_output_root', default='./resultAA', help='')
        self.parser.add_argument('--enc_in', type=int, default=264, help='encoder input size') 
        
        
        
        self.parser.add_argument('--D', type=int, default=8)
        self.parser.add_argument('--K1', type=int, default=3)
        self.parser.add_argument('--S', type=int, default=3)
        self.parser.add_argument('--K2', type=int, default=3)
        self.parser.add_argument('--K3', type=int, default=3)
        self.parser.add_argument('--K4', type=int, default=3)
       
        self.parser.add_argument('--num_layers', type=int, default=6)

        self.parser.add_argument('--d_model', type=int, default=16)  # 16
        self.parser.add_argument('--head', type=int, default=4)  # 4

    
        self.parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
        
        
        # optimization
        self.parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
        self.parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
        self.parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')  # 20
        self.parser.add_argument('--patience', type=int, default=20, help='early stopping patience') # 3  20
        self.parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer initial learning rate')# 5e-5
        self.parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate')


        self.parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=True)

        self.parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        self.parser.add_argument('--gpu', type=int, default=3, help='gpu')
        self.parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        self.parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')



    def parse(self):
        args = self.parser.parse_args()
        return args

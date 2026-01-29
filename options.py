import argparse
import torch


class Options(object):

    def __init__(self):
        
        self.parser = argparse.ArgumentParser(description='')

        self.parser.add_argument('--seed', type=int, default=0)
        self.parser.add_argument('--fold', type=int, default=10)
        # basic config       
        self.parser.add_argument('--model', type=str, default='Remove3') #BSGN  Ablation6  BSGN

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


        # self.parser.add_argument('--embed', type=str, default='fixed', help='time features encoding, options:[timeF, fixed, learned]')
        # self.parser.add_argument('--freq', type=str, default='s',
        #                 help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
        #                      'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        # self.parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers  (N)') # 3
        # self.parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        # self.parser.add_argument('--factor', type=int, default=1, help='attn factor')
        # self.parser.add_argument('--activation', type=str, default='gelu', help='activation')
        # self.parser.add_argument('--n_heads', type=int, default=4, help='num of heads, no use for WITRAN') 
        # self.parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn, no use for WITRAN') # 4* d_model 



    
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


        # others
        # self.parser.add_argument('--d_model', type=int, default=128, help='dimension of model hidden states (d_model)')
        # self.parser.add_argument('--n_heads', type=int, default=4, help='num of heads, no use for WITRAN') 
        # self.parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers  (N)') # 3
        # self.parser.add_argument('--d_layers', type=int, default=3, help='num of decoder layers, no use for WITRAN') #  
        # self.parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn, no use for WITRAN') # 4* d_model 

        # # For TimesNet
        # self.parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
        # self.parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')


    def parse(self):
        args = self.parser.parse_args()
        return args

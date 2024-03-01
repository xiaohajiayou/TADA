import argparse
# import matplotlib.pyplot as plt
# import numpy as np
import numpy as np
import torch
from torch.utils.data import DataLoader

from Informer import informer
from dataset import TRANSFORMERDATA
from model import *
import os
import random
import pandas as pd
# from sklearn.manifold import TSNE
dict = []

def score(pred, truth):
    """input must be tensors!"""
    x = pred-truth
    score1 = torch.tensor([torch.exp(-i/13)-1 for i in x if i<0])
    score2 = torch.tensor([torch.exp(i/10)-1 for i in x if i>=0])
    return int(torch.sum(score1)+torch.sum(score2))


def get_pred_result(data_len, out, lb):
    pred_sum, pred_cnt = torch.zeros(800), torch.zeros(800)
    for j in range(data_len):
        if j < seq_len-1:
            pred_sum[:j+1] += out[j, -(j+1):]
            pred_cnt[:j+1] += 1
        elif j <= data_len-seq_len:
            pred_sum[j-seq_len+1:j+1] += out[j]
            pred_cnt[j-seq_len+1:j+1] += 1
        else:
            pred_sum[data_len-seq_len+1-(data_len-j):data_len-seq_len+1] += out[j, :(data_len-j)]
            pred_cnt[data_len-seq_len+1-(data_len-j):data_len-seq_len+1] += 1
    truth = torch.tensor([lb[j,-1] for j in range(len(lb)-seq_len+1)], dtype=torch.float)
    pred_sum, pred_cnt = pred_sum[:data_len-seq_len+1], pred_cnt[:data_len-seq_len+1]
    pred2 = pred_sum/pred_cnt
    pred2 *= Rc
    truth *= Rc
    return truth, pred2 


def test():
    truth, tot, tot_sc = [], 0, 0

    net.eval()
    # s_model.eval()
    # t_model.eval()
    with torch.no_grad():
        for k in range(test_len):

            i = next(list_iter, None)
            if i is not None:

                dataset = TRANSFORMERDATA(i, seq_len)
                data_len = len(dataset)
                dataloader = DataLoader(dataset, batch_size=800, shuffle=0)
                it = iter(dataloader)
                d = next(it)
                input, lb, msk = d[0], d[1], d[2]
                if fake:
                    input = torch.zeros(input.shape)
                input, msk = input.cuda(), msk.cuda()
                # input, msk = input, msk
                #uncertainty(input, msk, data_len, lb, i)
                _, out,_ = net(input, msk)
                out = out.squeeze(2).cpu()
                truth, pred = get_pred_result(data_len, out, lb)

                # save rul
                data_path = '/home/user/Desktop/TADA/visual/rul/' \
                            + src_id + '-' + tgt_id+ '/' + i[6:]
                if not os.path.exists(data_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                    os.makedirs(data_path)


                # np.savetxt(data_path + '/after_da_rul.csv', pred, fmt='%.2f', delimiter=',')
                # np.savetxt(data_path + '/source_only.csv', pred, fmt='%.2f', delimiter=',')
                # np.savetxt(data_path + '/truth_rul.csv', truth, fmt='%.2f', delimiter=',')
                np.savetxt(data_path + '/im.csv', pred, fmt='%.2f', delimiter=',')
                # save rul
                mse = float(torch.sum(torch.pow(pred-truth, 2)))
                rmse = math.sqrt(mse/data_len)
                tot += rmse
                sc = score(pred, truth)
                tot_sc += sc
                print("for file {}: rmse={:.4f}, score={}".format(i, rmse, sc))
                dict.append(str(src_id)+'-'+str(i) +'-rmse:'+str(rmse)+'-score:'+ str(sc))
                np.savetxt('/home/user/Desktop/TADA/visual/rul/aesc.txt', dict,delimiter=',', fmt = '%s')
                print('-'*80)
           
    print("tested on [{}] files, mean RMSE = {:.4f}, mean score = {}".format(test_len, tot/test_len, int(tot_sc/test_len)))
    return tot/test_len, int(tot_sc/test_len)

if __name__ == "__main__":

    df = pd.DataFrame()
    res = []
    full_res = []
    print('=' * 89)

    print('=' * 89)
    av_loss = 0
    av_score=0

    for src_id in ['FD001', 'FD002', 'FD003', 'FD004']:
        for tgt_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    # for src_id in ['FD001']:
    #     for tgt_id in ['FD002']:
            if src_id != tgt_id:
                total_loss = []
                total_score = []
                Rc = 130
                fake = 0
                parser = argparse.ArgumentParser()
                parser.add_argument('--gpu_id', type=str, default='0')
                parser.add_argument("--seq_len", type=int, default=70)
                parser.add_argument("--source", type=str, default=src_id, help="file name the model trained on")
                parser.add_argument("--target", type=str, default=tgt_id, help="test domain")
                parser.add_argument("--sem", type=int, default=1)
                args = parser.parse_args()
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
                seq_len = args.seq_len
                net = mymodel(max_len=seq_len, dropout=0.5).cuda()
                # net = informer(96, False, 0, 24, 'timeF', 0.1, 24, 'h', 2, 3, 24, 8, True, 1, 24).cuda()
                model_name = args.source
                test_name = args.target
                new = 'both_'+args.source[-1]+args.target[-1]
                # 刚训练的网络权重保存
                dir = f"/home/user/Desktop/TADA/online/{src_id[-1]}{tgt_id[-1]}_net.pth"
                # dir = f"/home/user/Desktop/TADA/trained_model/im+trans+conv1024/{src_id[-1]}{tgt_id[-1]}_net.pth"

                # x=torch.load("save/final/"+new+".pth", map_location='cuda:0')
                x = torch.load(dir, map_location='cuda:0')

                net.load_state_dict(x)
                data_root = "CMAPSS/units/"
                label_root = "CMAPSS/labels/"
                lis = os.listdir(data_root)
                test_list = [i for i in lis if i[:5] == test_name]
                random.shuffle(test_list)
                test_len = len(test_list)
                list_iter = iter(test_list)
                # s_model, t_model = mymodel(max_len=seq_len).cuda(), mymodel(max_len=seq_len).cuda()
                # # s_model, t_model = mymodel(max_len=seq_len), mymodel(max_len=seq_len)
                # s_pth = "save/final/FD00"+new[-2]+"new.pth"
                # t_pth = "save/final/FD00"+new[-1]+"new.pth"
                # s_model.load_state_dict(torch.load(s_pth, map_location='cuda:0'))
                # t_model.load_state_dict(torch.load(t_pth, map_location='cuda:0'))
                test_loss, test_score = test()
                src_only_loss, src_only_score = test()
                total_loss.append(test_loss)
                total_score.append(test_score)
                loss_mean, loss_std = np.mean(np.array(total_loss)), np.std(np.array(total_loss))
                score_mean, score_std = np.mean(np.array(total_score)), np.std(np.array(total_score))
                full_res.append((f'{src_id}-->{tgt_id}', f'{src_only_loss:2.2f}',
                                 f'{loss_mean:2.2f}', f'{loss_std:2.2f}', f'{src_only_score:2.2f}',
                                 f'{score_mean:2.2f}', f'{score_std:2.2f}'))
                av_loss += loss_mean
                av_score += score_mean

    full_res.append(('average', ' ',
                     f'{av_loss / 12:2.2f}', ' ', ' ', f'{av_score / 12:2.2f}', ' '))
    df = df.append(pd.Series(('TADA')), ignore_index=True)
    df = df.append(pd.Series(('scenario', 'src_only_loss', 'mean_loss', 'std_loss', 'src_only_score',
                              f'mean_score', f'std_score')), ignore_index=True)
    df = df.append(pd.DataFrame(full_res), ignore_index=True)
    print('=' * 89)
    print(f'Results using: TADA')
    print('=' * 89)
    print(df.to_string())
    dir = '/home/user/Desktop/TADA/result'
    df.to_csv(f'{dir}/第三章+频域特征+相互注意力(不cat)+1000epoch.csv')



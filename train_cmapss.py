import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from Informer import informer
from functional import dwt, DWT1D
from model import *
import torch
# import numpy as np
# import matplotlib.pyplot as plt
import os
import random
import argparse
from dataset import TRANSFORMER_ALL_DATA, TRANSFORMERDATA
from torch.utils.data import DataLoader, random_split
from loss import advLoss, im, entropy, entropy_advLoss
import itertools
import time
import errno
import os
import os.path as osp

def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
def init_writer(log_dir):
    print(f"Initialize tensorboard (log_dir={log_dir})")
    writer = SummaryWriter(log_dir=log_dir)
    return writer
def write_scalar(writer,tag, scalar_value, global_step=None):
    if writer is None:
        # Do nothing if writer is not initialized
        # Note that writer is only used when training is needed
        pass
    else:
        writer.add_scalar(tag, scalar_value, global_step)

def validate():
    net.eval()
    tot = 0
    tot_score =0
    with torch.no_grad():
        for i in target_test_names:
            pred_sum, pred_cnt = torch.zeros(800), torch.zeros(800)
            valid_data = TRANSFORMERDATA(i, seq_len)
            data_len = len(valid_data)
            valid_loader = DataLoader(valid_data, batch_size=1000)
            valid_iter = iter(valid_loader)
            d = next(valid_iter)
            input, lbl, msk = d[0], d[1], d[2]
            input, msk = input.cuda(), msk.cuda()
            t_features, out,conv_fea = net(input, msk)
            out = out.squeeze(2).cpu()
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
            truth = torch.tensor([lbl[j,-1] for j in range(len(lbl)-seq_len+1)], dtype=torch.float)
            pred_sum, pred_cnt = pred_sum[:data_len-seq_len+1], pred_cnt[:data_len-seq_len+1]
            pred = pred_sum/pred_cnt
            mse = float(torch.sum(torch.pow(pred-truth, 2)))
            rmse = math.sqrt(mse/data_len)
            tot += rmse
            score = get_score(pred,truth)
            tot_score +=score
            rmse = tot * Rc / len(valid_list)
            score = tot_score * Rc / len(valid_list)
    return rmse,score

en = 240
def train():
    minn = 999
    writer_dir = (
        f'/home/user/Desktop/TADA/tensorboard/{args.source}to{args.target}')

    mkdir_if_missing(writer_dir)
    writer = init_writer(writer_dir)
    for e in range(epochs):
        al, tot = 0, 0
        net.train()
        random.shuffle(source_list)
        random.shuffle(target_list)
        source_iter, target_iter = iter(source_list), iter(target_list)
        loss2_sum, loss1_sum = 0, 0
        bkb_sum, out_sum,im_sum,all_sum,trans_sum = 0, 0,0,0,0
        patch_sum = 0
        cnt = 0
        s_iter = iter(DataLoader(s_data, batch_size=args.batch_size, shuffle=True))
        t_iter = iter(DataLoader(t_data, batch_size=args.batch_size, shuffle=True))
        l = min(len(s_iter), len(t_iter))
        for _ in range(l):
            s_d, t_d = next(s_iter), next(t_iter)
            s_input, s_lb, s_msk = s_d[0], s_d[1], s_d[2]
            t_input, t_msk = t_d[0], t_d[2]
            s_input, s_lb, s_msk = s_input.cuda(), s_lb.cuda(), s_msk.cuda()
            t_input, t_msk = t_input.cuda(), t_msk.cuda()
            # s_input, s_lb, s_msk = s_input, s_lb, s_msk
            # t_input, t_msk = t_input, t_msk

            # output1, rul, x
            s_features, s_out,s_conv_fea = net(s_input, s_msk)
            t_features, t_out,t_conv_fea = net(t_input, t_msk) # [bts, seq_len, feature_num]

            # 频域特征提取
            coefs = dwt(s_input, "haar")
            coeft = dwt(t_input, "haar")
            k = 8
            lohi = torch.zeros(2,k)
            lohi[0,:k//2] = 1.0/(k//2)**0.5
            lohi[1,k//2:] = 1.0/(k//2)**0.5*torch.arange(1,k//2+1)

            dwt_layer = DWT1D.apply
            pin_fea = dwt_layer(s_input,lohi.cuda())
            pin_fea = pin_fea.reshape(pin_fea.shape[0],pin_fea.shape[2],-1)
            s_features_ad = s_features.reshape(s_features.shape[0],s_features.shape[2],s_features.shape[1])
            # 128*24*146
            s_p_t_feature = torch.concat([pin_fea,s_features_ad],dim=-1)


            pin_fea = dwt_layer(t_input,lohi.cuda())
            pin_fea = pin_fea.reshape(pin_fea.shape[0],pin_fea.shape[2],-1)
            t_features_ad = t_features.reshape(t_features.shape[0],t_features.shape[2],t_features.shape[1])
            # 128*24*146
            t_p_t_feature = torch.concat([pin_fea,t_features_ad],dim=-1)

            s_out.squeeze_(2)
            t_out.squeeze_(2)
            loss1 = Loss(s_out, s_lb)
            loss1_sum += loss1
            cnt += 1
            if args.type == 1 or args.type == 0:
                if args.type == 1:
                    s_domain = D2(s_features)
                    t_domain = D2(t_features)
                else:
                    s_domain = D1(s_out)
                    t_domain = D1(t_out)
                loss2 = advLoss(s_domain.squeeze(1), t_domain.squeeze(1), 'cuda')
                loss2_sum += loss2
                loss = loss1 + a*loss2
            elif args.type == 2:
                # s_domain_bkb = D2(s_features)
                # t_domain_bkb = D2(t_features)
                # 用transformer+cnn
                # s_domain_bkb = D2(s_conv_fea)
                # t_domain_bkb = D2(t_conv_fea)
                # 用transformer+pin
                s_domain_bkb = D2(s_p_t_feature)
                t_domain_bkb = D2(t_p_t_feature)
                # D1为rul级别的鉴别器
                s_domain_out = D1(s_out)
                t_domain_out = D1(t_out)
                # # D3为时间步级别鉴别器
                # pat_s_domain = D3(s_features)
                # pat_t_domain = D3(t_features)
                if e>=5:

                    fea_loss = advLoss(s_domain_bkb.squeeze(1), t_domain_bkb.squeeze(1), 'cuda')
                    out_loss = advLoss(s_domain_out.squeeze(1), t_domain_out.squeeze(1), 'cuda')
                    # 基本域鉴别器损失总和
                    bkb_sum += fea_loss


                    loss = loss1 + 0.1*fea_loss+0.5*out_loss

                    all_sum+=loss
                else:
                    loss = loss1
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(net.parameters(), D1.parameters(), D2.parameters()), 2)
            opt.step()    

        rmse,score = validate()
        if args.type == 2:
            # if(e%10==1):
            print("FD00{}--->FD00{}||{}/{}| loss1={:.5f}, fea_loss={:.5f},\n all_loss={:.5f}, rmse={:.5f}, score={:.5f}".\
            format(source[-1],target[-1],e, args.epoch, loss1_sum/cnt, bkb_sum/cnt,all_sum/cnt, rmse,score))
            write_scalar(writer, "valid/rmse", rmse, e)
            write_scalar(writer, "valid/score", score, e)
            write_scalar(writer, "train/source_loss", loss1_sum/cnt, e)
            write_scalar(writer, "train/domain_da_loss", bkb_sum/cnt, e)
            write_scalar(writer, "train/all_loss", all_sum / cnt, e)


        else:    
            print("{}/{}| 1={:.5f}, 2={:.5f}, rmse={:.5f}".format(e, args.epoch, loss1, loss2_sum/cnt, rmse))
        if rmse<minn:
            minn = rmse
            print("min={}".format(minn))
            if args.type == 1:
                torch.save(net.state_dict(), "save/final/dann_"+source[-1]+target[-1]+".pth")
            elif args.type == 0:
                torch.save(net.state_dict(), "save/final/out_"+source[-1]+target[-1]+".pth")
            elif args.type == 2 :
                #torch.save(net.state_dict(), "save/final/both_"+source[-1]+target[-1]+".pth")
                # 保存寿命预测网络的权重到online文件夹
                torch.save(net.state_dict(), "online/"+source[-1]+target[-1]+"_net.pth")
                # torch.save(D1.state_dict(), "online/"+source[-1]+target[-1]+"_D1.pth")
                # torch.save(D2.state_dict(), "online/"+source[-1]+target[-1]+"_D2.pth")
                feature_path = '/home/user/Desktop/TADA/visual/feature/' \
                               + src_id + '-' + tgt_id
                if not os.path.exists(feature_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                    os.makedirs(feature_path)

                np.save(feature_path + '/' + str(rmse) + '_rme_' + str(score) + '_score' + '.npy',
                        t_features.cpu().detach().numpy())
        
        if args.scheduler:
            sch.step()

    return minn


def get_score(pred, truth):
    """input must be tensors!"""
    x = pred-truth
    score1 = torch.tensor([torch.exp(-i/13)-1 for i in x if i<0])
    score2 = torch.tensor([torch.exp(i/10)-1 for i in x if i>=0])
    return int(torch.sum(score1)+torch.sum(score2))


if __name__ == "__main__":
    for src_id in ['FD001', 'FD002', 'FD003', 'FD004']:
        for tgt_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    # for src_id in ['FD001']:
    #     for tgt_id in ['FD003']:
    # for src_id in ['FD001']:
    #     for tgt_id in ['FD002', 'FD003', 'FD004']:
            if src_id != tgt_id:
                seed = 0
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                random.seed(seed)
                np.random.seed(seed)
                Rc = 130

                parser = argparse.ArgumentParser()
                parser.add_argument('--gpu', type=str, default='0')
                parser.add_argument('--lr', type=float, default=0.02)#0.02 best 38.96 1422  0.05好像有用
                parser.add_argument("--epoch", type=int, default=en)
                parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
                parser.add_argument("--seq_len", type=int, default=70)
                parser.add_argument("--source", type=str, default=src_id, help="decide source file", choices=['FD001','FD002','FD003','FD004'])
                parser.add_argument("--target", type=str, default=tgt_id, help="decide target file", choices=['FD001','FD002','FD003','FD004'])
                parser.add_argument("--a", type=float, default=0.1, help='hyper-param α')
                parser.add_argument("--b", type=float, default=0.5, help='hyper-param β')
                parser.add_argument("--scheduler", type=int, default=1, choices=[0,1], help="1 for using sheduler while 0 for not")
                parser.add_argument("--type", type=int, default=2, choices=[0,1,2], help="0:out only | 1:DANN | 2:backbone+output")
                parser.add_argument("--train", default=1, type=int)
                args = parser.parse_args()
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
                source, target = args.source, args.target
                data_root = "CMAPSS/units/"
                label_root = "CMAPSS/labels/"
                type = {0:"out_only", 1:"DANN", 2:"backbone + output"}
                seq_len, a, epochs, b = args.seq_len, args.a, args.epoch, args.b
                option_str = "source={}, target={}, a={}, b={}, epochs={}, type={}, lr={}, {}using scheduler".\
                    format(source, target, a, b, epochs, type[args.type], args.lr, "" if args.scheduler else "not ")
                print(option_str)

                net = mymodel(max_len=seq_len)
                # net = informer(96,False,0,24,'timeF',0.1,24,'h',2,3,24,8,True,1,24)
                D1 = Discriminator(seq_len)
                D2 = backboneDiscriminator(seq_len)
                # D1 = Discriminator(36)

                # D2 = backboneDiscriminator(6)
                D3 = patch_backboneDiscriminator(70)
                if args.type == 0:
                    opt = torch.optim.SGD(itertools.chain(net.parameters(), D1.parameters()), lr=args.lr)
                elif args.type == 1:
                    opt = torch.optim.SGD(itertools.chain(net.parameters(), D2.parameters()), lr=args.lr)
                elif args.type == 2:
                    opt = torch.optim.SGD(itertools.chain(net.parameters(), D1.parameters(), D2.parameters()), lr=args.lr)
                Loss = nn.MSELoss()
                net, Loss, D1, D2,D3 = net.cuda(), Loss.cuda(), D1.cuda(), D2.cuda(),D3.cuda()
                # net, Loss, D1, D2 = net, Loss, D1, D2
                sch = torch.optim.lr_scheduler.StepLR(opt, 80, 0.5)

                source_list = np.loadtxt("save/"+source+"/train"+source+".txt", dtype=str).tolist()
                target_list = np.loadtxt("save/"+target+"/train"+target+".txt", dtype=str).tolist()
                valid_list = np.loadtxt("save/"+target+"/test"+target+".txt", dtype=str).tolist()
                a_list = np.loadtxt("save/"+target+"/valid"+target+".txt", dtype=str).tolist()
                target_test_names = valid_list + a_list
                minl = min(len(source_list), len(target_list))
                s_data = TRANSFORMER_ALL_DATA(source_list, seq_len)
                t_data = TRANSFORMER_ALL_DATA(target_list, seq_len)
                t_data_test = TRANSFORMER_ALL_DATA(target_test_names, seq_len)
                if not os.path.exists('trained_model/im+trans+conv2'):
                    os.makedirs('trained_model/im+trans+conv2')

                if args.train:
                    train_time1 = time.perf_counter()
                    minn = train()
                    train_time2 = time.perf_counter()
                    print(option_str)
                    print("best = {}, train time = {}".format(minn, train_time2-train_time1))



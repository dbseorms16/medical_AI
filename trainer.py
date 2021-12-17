import torch
import utility
from decimal import Decimal
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd 
from pandas import DataFrame
import numpy as np


class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.error_last = 1e8
        self.epoch = 0


    def loadLabel(self, filenames ):
        label=[]
        df = pd.read_csv("./labels.csv", dtype='unicode')
        for filename in filenames:
            k = int(filename.split('\\')[-2])
            label.append(int(df[df['PatientID'] == str(k)]['ClassLabel']))
        return torch.tensor(label).to('cuda:0')

    def train(self):
        self.epoch += 1

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate:'.format(self.epoch)
        )
        self.loss.start_log()
        self.model.train()
        criterion = nn.BCEWithLogitsLoss()
        losses = 0
        # for p, n in self.model.named_parameters():
        #     print(n.requires_grad)
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (hr, filename) in enumerate(self.loader_train):
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()
            gt = self.loadLabel(filename)
            gt = nn.functional.one_hot(gt, num_classes=2).type(torch.float)
            
            # forward
            result = self.model(hr.to('cuda:0'))
            # gt = torch.ones((16,2)).type(torch.float).to('cuda:0')

            # compute primary loss
            loss = criterion(result, gt)


            loss.backward()                
            self.optimizer.step()
            self.optimizer.zero_grad()
                
            timer_model.hold()
            losses = loss
            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    losses,
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()

        timer_test = utility.timer()
        score = 0
        with torch.no_grad():
            scale = max(self.scale)
            for si, s in enumerate([scale]):
                eval_psnr = 0
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for _, (hr, filename) in enumerate(tqdm_test):

                    gt = self.loadLabel(filename)
                    # gt = nn.functional.one_hot(gt, num_classes=2).type(torch.float)

                    results = self.model(hr.to('cuda:0')).squeeze(0)
                    r_i = F.softmax(results, dim=0).argmax()
                    # result = r_i[r_i.argmax()]
                    # if r_i == gt  ,.argmax()
                    if r_i == gt:
                        score += 1
                    # print(gt)

                self.ckp.log[-1, si] = (score* 100) / len(self.loader_test) 
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{}]\tACCU: {:.2f} (Best: {:.2f} @epoch {})'.format(
                        self.opt.data_test,
                        self.ckp.log[-1, si],
                        best[0][si],
                        best[1][si] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.opt.test_only:
            self.ckp.save(self, self.epoch, is_best=(best[1][0] + 1 == self.epoch))

    def step(self):
        self.scheduler.step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        if len(args)>1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]], 

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs

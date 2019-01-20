import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from sklearn.metrics import f1_score
import warnings
import shutil
import time
import subprocess
import sys
from utils import f1_metric, print_duration, get_hash, str_date_time, dict_to_csv, save_plot, copy_files, save_plots


class Learner:

    def __init__(self, model, dataloaders, loss_func, optimizer, scheduler, args):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.recorder = Recorder(args)
        self.args = args

        if len(dataloaders) == 3:
            self.train_dl, self.val_dl, self.test_dl = dataloaders
        elif len(dataloaders) == 2:
            self.train_dl, self.val_dl = dataloaders
            self.test_dl = None
        elif len(dataloaders) == 1:
            self.train_dl = dataloaders
            self.val_dl = self.test_dl = None

    @staticmethod
    def to_cuda(data):
        if type(data) == tuple:
            return [tensor.cuda() for tensor in data]
        else:
            return [data.cuda(), None]

    def fit(self, epoch, n_eval, tresh, early_stop, warmup_epoch, clip):

        step = 0
        min_loss = 1e5
        max_f1 = 0
        max_test_f1 = 0
        no_improve_epoch = 0
        no_improve_in_previous_epoch = False
        fine_tuning = False
        losses = []
        best_test_info = None
        torch.backends.cudnn.benchmark = False
        eval_every = int(len(list(iter(self.train_dl))) / n_eval)

        time_start = time.time()
        for e in range(epoch):
            self.scheduler.step()
            if e >= warmup_epoch:
                if no_improve_in_previous_epoch:
                    no_improve_epoch += 1
                    if no_improve_epoch >= early_stop:
                        e = e-1
                        break
                else:
                    no_improve_epoch = 0
                no_improve_in_previous_epoch = True
            if not fine_tuning and e >= warmup_epoch:
                self.model.embedding.weight.requires_grad = True
                fine_tuning = True
            self.train_dl.init_epoch()

            for train_batch in iter(self.train_dl):
                step += 1
                self.model.zero_grad()
                self.model.train()
                model_input = self.to_cuda(train_batch.text)
                y = train_batch.target.type(torch.Tensor).cuda()
                pred = self.model.forward(*model_input).view(-1)
                loss = self.loss_func(pred, y)
                self.recorder.tr_record.append({'tr_loss': loss.cpu().data.numpy()})
                loss.backward()
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                # parameters = list(filter(lambda p: p.grad is not None, self.model.parameters()))
                # total_norm = 0
                # for p in parameters:
                #     param_norm = p.grad.data.norm(2)
                #     total_norm += param_norm.item() ** 2
                # total_norm = total_norm ** (1. / 2)

                self.recorder.norm_record.append({'grad_norm': total_norm})
                self.optimizer.step()

                if step % eval_every == 0:
                    with torch.no_grad():
                        # train evaluation
                        losses.append(loss.cpu().data.numpy())
                        train_loss = np.mean(losses)

                        # val evaluation
                        val_loss, val_f1, val_ids, val_prob, val_true = self.evaluate(self.val_dl, tresh)
                        val_pred_to_csv(val_ids, val_prob, val_true, f'val_probs_{int(step/eval_every) - 1}.csv')
                        self.recorder.val_record.append({'step': step, 'loss': val_loss, 'f1': val_f1})
                        info = {'best_ep': e, 'step': step, 'train_loss': train_loss,
                                'val_loss': val_loss, 'val_f1': val_f1}
                        self.recorder.save_step(info, message=True)
                        if val_f1  > max_f1:
                            self.recorder.save(self.model, info)
                            max_f1 = val_f1
                            no_improve_in_previous_epoch = False
                        if val_loss < min_loss:
                            min_loss = val_loss

                        # test evaluation
                        # if self.args.test:
                        #     test_loss, test_f1 =  self.evaluate(self.test_dl, tresh)
                        #    test_info = {'test_ep': e, 'test_step': step, 'test_loss': test_loss, 'test_f1': test_f1}
                        #    self.recorder.test_record.append({'step': step, 'loss': test_loss, 'f1': test_f1})
                        #     print('epoch {:02} - step {:06} - test_loss {:.4f} - test_f1 {:.4f}'.format(*list(test_info.values())))
                         #   if test_f1 >= max_test_f1:
                         #       max_test_f1 = test_f1
                         #       best_test_info = test_info

        tr_time = print_duration(time_start, 'training time: ')
        self.recorder.append_info({'ep_time': tr_time/(e + 1)})

        #if self.args.test:
        #    self.recorder.append_info(best_test_info, message='Best results for test:')
        self.recorder.append_info({'min_loss': min_loss}, 'min val loss: ')

        self.model, info = self.recorder.load(message = 'best model:')

        # final train evaluation
        train_loss, train_f1, _, _, _ = self.evaluate(self.train_dl, tresh)
        tr_info = {'train_loss':train_loss, 'train_f1':train_f1}
        self.recorder.append_info(tr_info, message='train loss and f1:')


    def evaluate(self, dl, tresh):
        with torch.no_grad():
            dl.init_epoch()
            self.model.cell.flatten_parameters()
            self.model.eval()
            self.model.zero_grad()
            loss = []
            tp = 0
            n_targs = 0
            n_preds = 0
            probs = []
            targs = []
            ids = []
            for batch in iter(dl):
                model_input = self.to_cuda(batch.text)
                y = batch.target.type(torch.Tensor).cuda()
                pred = self.model.forward(*model_input).view(-1)
                loss.append(self.loss_func(pred, y).cpu().data.numpy())
                prob = torch.sigmoid(pred).cpu().data.numpy()
                label = (prob > tresh).astype(int)
                y = y.cpu().data.numpy()
                tp += sum(y + label == 2)
                n_targs += sum(y)
                n_preds += sum(label)
                probs += prob.tolist()
                targs += y.tolist()
                ids += batch.qid.view(-1).data.numpy().tolist()
            f1 = f1_metric(tp, n_targs, n_preds)
            loss = np.mean(loss)
        return loss, f1, ids, probs, targs

    def predict_probs(self, is_test=False):
        if is_test:
            print('predicting test dataset...')
            dl = self.test_dl
        else:
            print('predicting validation dataset...')
            dl = self.val_dl

        self.model.cell.flatten_parameters()
        self.model.eval()
        y_pred = []
        y_true = []
        ids = []

        dl.init_epoch()
        for batch in iter(dl):
            model_input = self.to_cuda(batch.text)
            if not is_test or self.args.test:
                y_true += batch.target.data.numpy().tolist()
            y_pred += torch.sigmoid(self.model.forward(*model_input).view(-1)).cpu().data.numpy().tolist()
            ids += batch.qid.view(-1).data.numpy().tolist()
        return y_pred, y_true, ids

    def predict_labels(self, is_test=False, thresh=0.5):
        y_prob, y_true, ids = self.predict_probs(is_test=is_test)

        if type(thresh) == list:
            thresh, max_f1 = choose_thresh(y_prob, y_true, *thresh, message=True)
            self.recorder.append_info({'best_tr': thresh, 'best_f1': max_f1})

        y_label = (np.array(y_prob) >= thresh).astype(int)
        return y_label, y_prob, y_true, ids, thresh





class Recorder:

    models_dir = './models'
    best_model_path = './models/best_model.m'
    best_info_path = './models/best_model.info'
    record_dir = './notes'

    def __init__(self, args):
        self.args = args
        if self.args.mode == 'test':
            self.record_path = './notes/test_records.csv'
        elif self.args.mode == 'run':
            self.record_path = './notes/records.csv'
        self.val_record = []
        self.tr_record = []
        self.test_record = []
        self.norm_record = []
        self.new = True

    @classmethod
    def append_info(cls, dict, message=None):
        dict = format_info(dict)
        if message:
            print(message, dict)
        info = torch.load(cls.best_info_path)
        info.update(dict)
        torch.save(info, cls.best_info_path)

    def save(self, model, info):
        os.makedirs(self.models_dir, exist_ok=True)
        torch.save(model, self.best_model_path)
        info = format_info(info)
        torch.save(info, self.best_info_path)

    def load(self, message=None):
        model = torch.load(self.best_model_path)
        info = torch.load(self.best_info_path)
        if message:
            print(message, info)
        return model, info

    def save_step(self, step_info, message = False):
        if self.new:
            header = True
            mode = 'w'
            self.new = False
        else:
            header=False
            mode = 'a'
        dict_to_csv(step_info, 'train_steps.csv', mode, orient='columns', header=header)
        if message:
            print('epoch {:02} - step {:06} - train_loss {:.4f} - val_loss {:.4f} - f1 {:.4f}'.format(
                *list(step_info.values())))

    def record(self, fold):
        # save plots
        save_plot(self.val_record, 'loss', self.args.n_eval, 'val_loss')
        save_plot(self.val_record, 'f1', self.args.n_eval, 'val_f1')
        save_plot(self.norm_record, 'grad_norm', self.args.n_eval, 'grad_norm')
        if self.args.test:
            save_plots([self.val_record, self.test_record], ['loss', 'f1'], ['val', 'test'],self.args.n_eval)

        # create subdir for this experiment
        os.makedirs(self.record_dir, exist_ok=True)
        subdir = os.path.join(self.models_dir, str_date_time())
        if self.args.mode == 'test':
            subdir +=  '_test'
        os.mkdir(subdir)

        # write model params and results to csv
        csvlog = os.path.join(subdir, 'info.csv')
        param_dict = {}
        for arg in vars(self.args):
            param_dict[arg] = str(getattr(self.args, arg))
        info = torch.load(self.best_info_path)
        hash = get_hash() if self.args.machine == 'dt' else 'no_hash'
        passed_args = ' '.join(sys.argv[1:])
        param_dict = {'hash':hash, 'subdir':subdir, **param_dict, **info, 'args': passed_args}
        dict_to_csv(param_dict, csvlog, 'w', 'index', reverse=False)
        header = True if fold == 0 else False
        dict_to_csv(param_dict, self.record_path, 'a', 'columns', reverse=True, header=header)

        # copy all records to subdir
        png_files = ['val_loss.png', 'val_f1.png'] if not self.args.test else ['loss.png', 'f1.png']
        csv_files = ['val_probs*.csv', 'train_steps.csv', 'submission.csv', 'test_probs.csv']
        copy_files([*png_files, 'models/*.info', *csv_files], subdir)
        return subdir


def format_info(info):
    keys = list(info.keys())
    values = list(info.values())
    for k, v in zip(keys, values):
        info[k] = round(v, 4)
    return info


def choose_thresh(probs, true, thresh_range, message=True):
    min_th, max_th, th_step = thresh_range
    tmp = [0, 0, 0]  # idx, current_f1, max_f1
    th = min_th
    for tmp[0] in np.arange(min_th, max_th, th_step):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tmp[1] = f1_score(true, probs > tmp[0])
        if tmp[1] > tmp[2]:
            th = tmp[0]
            tmp[2] = tmp[1]
    if message:
        print('best threshold is {:.4f} with F1 score: {:.4f}'.format(th, tmp[2]))

    return th, tmp[2]


def val_pred_to_csv(ids, y_pred, y_true, fpath='val_probs.csv', mode='w'):
    df = pd.DataFrame()
    df['qid'] = ids
    df['prediction'] = y_pred
    df['true_label'] = y_true
    header = True if mode == 'w' else False
    df.to_csv(fpath, index=False, mode=mode, header=header)

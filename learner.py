import numpy as np
import torch
import os
from sklearn.metrics import f1_score
import warnings
import shutil
import time
import subprocess
import sys
from utils import f1_metric, print_duration, get_hash, str_date_time, dict_to_csv, save_plot, copy_files


class Learner:

    models_dir = './models'
    best_model_path = './models/best_model.m'
    best_info_path = './models/best_model.info'
    record_dir = './notes'

    def __init__(self, model, dataloaders, loss_func, optimizer, scheduler, args):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        if len(dataloaders) == 3:
            self.train_dl, self.val_dl, self.test_dl = dataloaders
        elif len(dataloaders) == 2:
            self.train_dl, self.val_dl = dataloaders
        elif len(dataloaders) == 1:
            self.train_dl = dataloaders
            self.val_dl = None
        self.args = args
        self.val_record = []
        self.train_record = []
        if self.args.mode == 'test':
            self.record_path = './notes/test_records.csv'
        elif self.args.mode == 'run':
            self.record_path = './notes/records.csv'

    def fit(self, epoch, eval_every, tresh, early_stop=1, warmup_epoch=2):
        print('Start training!')
        time_start = time.time()
        step = 0
        max_f1 = 0
        no_improve_epoch = 0
        no_improve_in_previous_epoch = False
        fine_tuning = False
        losses = []
        torch.backends.cudnn.benchmark = True

        for e in range(epoch):
            self.scheduler.step()
            if e >= warmup_epoch:
                if no_improve_in_previous_epoch:
                    no_improve_epoch += 1
                    if no_improve_epoch >= early_stop:
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
                x = train_batch.text.cuda()
                y = train_batch.target.type(torch.Tensor).cuda()
                pred = self.model.forward(x).view(-1)
                loss = self.loss_func(pred, y)
                self.train_record.append({'tr_loss': loss.cpu().data.numpy()})
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    losses.append(loss.cpu().data.numpy())
                    train_loss = np.mean(losses)
                    if step % eval_every == 0:
                        val_loss, val_f1 = self.evaluate(self.val_dl, tresh)
                        self.val_record.append({'step': step, 'loss': val_loss, 'val_f1': val_f1})
                        info = self.format_info(info = {'best_ep': e, 'step': step, 'train_loss': train_loss,
                                'val_loss': val_loss, 'val_f1': val_f1})
                        print('epoch {:02} - step {:06} - train_loss {:.4f} - val_loss {:.4f} - f1 {:.4f}'.format(
                            *list(info.values())))
                        if val_f1  >= max_f1:
                            self.save(info)
                            max_f1 = val_f1
                            no_improve_in_previous_epoch = False
        print_duration(time_start, 'Training time: ')

        # calculate train loss and train f1_score
        print('Evaluating model on train dataset')
        train_loss, train_f1 = self.evaluate(self.train_dl, tresh)
        tr_info = self.format_info({'train_loss':train_loss, 'train_f1':train_f1})
        print(tr_info)
        self.append_info(tr_info)

        m_info = self.load()
        print(f'Best model: {m_info}')

    def evaluate(self, dl, tresh):
        with torch.no_grad():
            self.train_dl.init_epoch()
            self.model.eval()
            self.model.zero_grad()
            loss = []
            tp = 0
            n_targs = 0
            n_preds = 0
            for batch in iter(dl):
                x = batch.text.cuda()
                y = batch.target.type(torch.Tensor).cuda()
                pred = self.model.forward(x).view(-1)
                loss.append(self.loss_func(pred, y).cpu().data.numpy())
                label = (torch.sigmoid(pred).cpu().data.numpy() > tresh).astype(int)
                y = y.cpu().data.numpy()
                tp += sum(y + label == 2)
                n_targs += sum(y)
                n_preds += sum(label)
            f1 = f1_metric(tp, n_targs, n_preds)
            loss = np.mean(loss)
        return loss, f1

    def predict_probs(self, is_test=False):
        if is_test:
            print('Predicting test dataset...')
            dl = self.test_dl
        else:
            print('Predicting validation dataset...')
            dl = self.val_dl

        self.model.lstm.flatten_parameters()
        self.model.eval()
        y_pred = []
        y_true = []
        ids = []

        dl.init_epoch()
        for batch in iter(dl):
            x = batch.text.cuda()
            if not is_test:
                y_true += batch.target.data.numpy().tolist()
            y_pred += torch.sigmoid(self.model.forward(x).view(-1)).cpu().data.numpy().tolist()
            ids += batch.qid.view(-1).data.numpy().tolist()
        return y_pred, y_true, ids

    def predict_labels(self, is_test=False, tresh=0.5):
        def _choose_tr(self, min_tr, max_tr, tr_step):
            print('Choosing treshold.\n')
            val_pred, val_true, _ = self.predict_probs(is_test=False)
            tmp = [0, 0, 0]  # idx, current_f1, max_f1
            tr = min_tr
            for tmp[0] in np.arange(min_tr, max_tr, tr_step):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    tmp[1] = f1_score(val_true, np.array(val_pred) > tmp[0])
                if tmp[1] > tmp[2]:
                    tr = tmp[0]
                    tmp[2] = tmp[1]
            print('Best threshold is {:.4f} with F1 score: {:.4f}'.format(tr, tmp[2]))
            return tr, tmp[2]

        y_pred, y_true, ids = self.predict_probs(is_test=is_test)

        if type(tresh) == list:
            tresh, max_f1 = _choose_tr(self, *tresh)
            self.append_info({'best_tr': tresh, 'best_f1': max_f1})

        y_label = (np.array(y_pred) >= tresh).astype(int)
        return y_label, y_true, ids, tresh

    def save(self, info):
        os.makedirs(self.models_dir, exist_ok=True)
        torch.save(self.model, self.best_model_path)
        torch.save(info, self.best_info_path)

    @staticmethod
    def format_info(info):
        keys = list(info.keys())
        values = list(info.values())
        for k, v in zip(keys, values):
            info[k] = round(v, 4)
        return info

    @classmethod
    def append_info(cls, dict):
        dict = cls.format_info(dict)
        info = torch.load(cls.best_info_path)
        info.update(dict)
        torch.save(info, cls.best_info_path)

    def record(self):
        # save plots
        save_plot(self.val_record, 'loss', self.args.n_eval)
        save_plot(self.val_record, 'val_f1', self.args.n_eval)
        save_plot(self.train_record, 'tr_loss', self.args.n_eval)

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
        hash = get_hash()
        passed_args = ' '.join(sys.argv[1:])
        param_dict = {'hash':hash, 'subdir':subdir, **param_dict, **info, 'args': passed_args}
        dict_to_csv(param_dict, csvlog, 'w', 'index', reverse=False)
        dict_to_csv(param_dict, self.record_path, 'a', 'columns', reverse=True)

        # copy all records to subdir
        copy_files(['*.png', 'models/*.m', 'models/*.info'], subdir)

    def load(self):
        self.model = torch.load(self.best_model_path)
        info = torch.load(self.best_info_path)
        return info

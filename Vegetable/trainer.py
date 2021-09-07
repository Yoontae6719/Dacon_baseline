# base library
from time import time
import numpy as np
import pandas as pd

# torch library
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

# custom lib
from conf.conf import Conf
from models.seq2seq import Encoder, Decoder,BahdanauAttention
from models.net import Seq2Seq
from models.utils import my_custom_metric
from data.V_dataLoader import VegetableDataloder, preprocessing
from progress_bar import ProgressBar

# etc
import neptune.new as neptune
import click

@click.command()
@click.option('--exp_name', type=str, default="dacon_data")
@click.option('--conf_file_path', type=str, default="./conf/dacon_data.yaml")
@click.option("--seed", type = int, default = None)
def main(exp_name, conf_file_path, seed):
    if exp_name is None:
        exp_name = click.prompt('▶ experiment name', default='default')

    log_each_step = True
    if '!' in exp_name:
        exp_name = exp_name.replace('!', '')
        log_each_step = False

    split = exp_name.split('@')
    if len(split) == 2:
        seed = int(split[1])
        exp_name = split[0]

    cnf = Conf(conf_file_path=conf_file_path,
               seed=seed,
               exp_name=exp_name,
               log=log_each_step)

    print(f'\n{cnf}')
    print(f'\n▶ Starting Experiment \'{exp_name}\' [seed: {cnf.seed}]')

    trainer = Trainer(cnf=cnf)
    trainer.run_model()
    trainer.submission()

class Trainer(object):
    def __init__(self, cnf):
        self.cnf = cnf

        print(self.cnf)


        self.run = neptune.init(
            api_token= self.cnf.api_token,
            project = self.cnf.neptune_path,
            source_files= ["triner.py"]
        )


        # init dataloader
        x_train, y_train, x_val, y_val = preprocessing(self.cnf)
        train_dataset = VegetableDataloder(x_train, y_train)
        val_dataset = VegetableDataloder(x_val, y_val)

        self.train_dataloader = DataLoader(dataset = train_dataset,
                                          batch_size= self.cnf.batch_size,
                                           num_workers= 16, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset,
                                           batch_size=self.cnf.batch_size,
                                           num_workers=16, shuffle=True)

        self.encoder = Encoder(input_dim=x_train.shape[-1],
                          hidden_dim=self.cnf.hidden_dim,
                          n_layers=self.cnf.hidden_layer,
                          dropout=self.cnf.dropout)

        self.attention = BahdanauAttention(dec_output_dim=self.cnf.hidden_dim,
                                      units=self.cnf.hidden_dim)
        self.decoder = Decoder(
            dec_feature_size=self.cnf.target_n,
            encoder_hidden_dim=self.cnf.hidden_dim, output_dim=self.cnf.target_n,
            decoder_hidden_dim=self.cnf.hidden_dim, n_layers=self.cnf.hidden_layer,
            dropout=self.cnf.dropout,
            attention=self.attention
        )
        self.Seq2Seq = Seq2Seq(self.encoder, self.decoder, self.attention, self.cnf.device)

        self.model = self.Seq2Seq.to(self.cnf.device)

        # init optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=cnf.lr)
        self.criterion = nn.L1Loss()  # mae
        self.custom_metric = my_custom_metric

        # set epoch
        self.epoch = 0

        # loss
        self.total_loss, self.total_score = 0, 0
        self.total_val_loss, self.total_val_score  = 0, 0

        self.loss_plot, self.score_plot = [], []
        self.val_loss_plot, self.val_score_plot = [], []

        # set log path
        self.log_path = cnf.exp_log_path

        # Set neptune
        self.run["config/dataset/path"] = "./data/private_data/private_data.csv"
        self.run["config/dataset/size"] = len(x_train)
        self.run["config/model"] = type(self.model).__name__
        self.run["config/criterion"] = type(self.criterion).__name__
        self.run["config/optimizer"] = type(self.optimizer).__name__
        self.run["config/hyperparameters"] = self.cnf.all_params

        # Progress bar
        self.log_freq = len(self.train_dataloader)
        self.progress_bar = ProgressBar(max_step=self.log_freq, max_epoch=self.cnf.epoch)

    def train(self):

        times = []
        self.total_loss, self.total_score = 0, 0
        self.total_val_loss, self.total_val_score  = 0, 0

        self.training = True
        for batch, batch_item in enumerate(self.train_dataloader):
            t1 = time()

            batch_loss, batch_score = self.train_step(batch_item = batch_item,
                                                      batch = batch,
                                                      training= self.training, teacher_forcing= False)

            self.total_loss += batch_loss
            self.total_score += batch_score

            times.append(time() - t1)

            print(f'\r{self.progress_bar} '
                      f'│ Loss: {(self.total_loss / (batch+1)):.6f} '
                      f'│ Loss: {(self.total_score / (batch+1)):.6f} '
                      f'│ ↯: {1 / np.mean(times):5.2f} step/s', end='')

            self.progress_bar.inc()


        self.run["training/batch/loss"].log(self.total_loss / (batch +1) )
        self.run["training/batch/score"].log(self.total_score / (batch +1) )

        self.loss_plot.append(self.total_loss / (batch + 1))
        self.score_plot.append(self.total_score / (batch + 1))

        self.training = False

        t2 = time()

        for val_batch, val_batch_item in enumerate(self.val_dataloader):
            batch_val_loss, batch_val_score = self.train_step(batch_item = val_batch_item,
                                                              batch = val_batch,
                                                              training= self.training, teacher_forcing= False)
            self.total_val_loss += batch_val_loss
            self.total_val_score += batch_val_score

        self.run["val/batch/loss"].log(self.total_val_loss / (val_batch +1) )
        self.run["val/batch/score"].log(self.total_val_score / (val_batch +1) )

        print("\n")
        print(f'\t● AVG Loss on Validation-set: {self.total_val_loss:.6f} │ T: {time() - t2:.2f} s')
        print(f'\t● AVG SMAPE on Validation-set: {self.total_val_score:.6f} │ T: {time() - t2:.2f} s')

        self.val_loss_plot.append(self.total_val_loss / (val_batch + 1))
        self.val_score_plot.append(self.total_val_score / (val_batch + 1))

        if np.min(self.val_loss_plot) == self.val_loss_plot[-1]:

            torch.save(self.model, f'./log/best_model.pth')
            torch.save(self.model.state_dict(), self.log_path / self.cnf.exp_name + f'_best_{self.val_loss_plot[-1]}.pth')
           # self.run[f"io_files/artifacts/seq2seq_arch"].upload(f"./_arch.txt")
            self.run[f"io_files/artifacts/seq2seq"].upload(f"./log/best_model.pth" )


    def train_step(self, batch_item, batch, training ,teacher_forcing):
        encoder_input = batch_item['encoder_input'].to(self.cnf.device)
        decoder_input = batch_item['decoder_input'].to(self.cnf.device)

        if training is True:
            self.model.train()
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = self.model(encoder_input, decoder_input, teacher_forcing)
                loss = self.criterion(output, decoder_input[:, 1:])
                score = self.custom_metric(output, decoder_input[:, 1:])

            loss.backward()
            self.optimizer.step()

            return loss, score
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(encoder_input, decoder_input, False)
                loss = self.criterion(output, decoder_input[:, 1:])
                score = self.custom_metric(output, decoder_input[:, 1:])
            return loss, score


    def inference(self, test_encoder):
        self.model = torch.load("./log/best_model.pth")
        self.model = self.model.to(self.cnf.device)

        self.model.train()
        encoder_input = test_encoder.to(self.cnf.device)
        decoder_input = torch.zeros([1, self.cnf.decoder_length + 1, self.cnf.target_n],
                                    dtype = torch.float32).to(self.cnf.device)
        with torch.no_grad():
            output = self.model(encoder_input, decoder_input, False)

        return output.cpu()

    def submission(self):
        data = pd.read_csv('./data/public_data/train.csv')

        week_day_map = {}
        for i, d in enumerate(data['요일'].unique()):
            week_day_map[d] = i
        data['요일'] = data['요일'].map(week_day_map)

        norm = data.iloc[:, 1:].max(0)
        submission = pd.read_csv("./data/sample_submission.csv")
        public_date_list = submission[submission['예측대상일자'].str.contains('2020')]['예측대상일자'].str.split('+').str[
            0].unique()


        # submission inference
        troch_norm = torch.tensor(norm.to_numpy()[2::2])

        for date in public_date_list:
            test_df = pd.read_csv(f'./data/public_data/test_files/test_{date}.csv')
            data = pd.read_csv('./data/public_data/train.csv')
            data = pd.concat([data, test_df]).iloc[-self.cnf.encoder_length:]

            week_day_map = {}
            for i, d in enumerate(data['요일'].unique()):
                week_day_map[d] = i
            data['요일'] = data['요일'].map(week_day_map)
            data = data.iloc[:, 1:] / norm

            encoder_input = torch.tensor(data.to_numpy(), dtype=torch.float32)
            encoder_input = encoder_input.unsqueeze(0)
            output = self.inference(encoder_input) * troch_norm

            idx = submission[submission['예측대상일자'].str.contains(date)].index
            submission.loc[idx, '배추_가격(원/kg)':] = output[0, [6, 13, 27]].numpy()
            submission.to_csv('dacon_baseline.csv', index=False)



    def save_ck(self):
        """
        save training checkpoint
        """
        ck = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_test_loss': self.total_val_loss
        }
        torch.save(ck, self.log_path / f'training_{self.epoch}.ck')


    def run_model(self):
        for _ in range(self.epoch, self.cnf.epoch):

            self.train()
            self.epoch +=1

            if self.epoch % 10:

                self.save_ck()

        self.run.stop()



if __name__ == '__main__':
    main()
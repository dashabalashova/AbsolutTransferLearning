import sys
sys.path.append('../Code')

import CNN_Lightning
import importlib
importlib.reload(CNN_Lightning)
from CNN_Lightning import CNNModel, AbLightDataset, f1
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
import os
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

torch.set_num_threads(25)

random_seed = 0
seed_everything(random_seed, workers=True)

def dx_f(r, random_seed = 0, k = 0):
    dx = AbLightDataset(k = k, r = r, batch_size = 16, random_seed = random_seed, data_f = 'data_Mason.tsv')
    dx.setup()
    return dx

os.makedirs('../Results/', exist_ok = True)

try:
    df2 = pd.read_csv('../Results/transf_metr_m4.tsv', sep='\t')
    run = int(df2.run.max()+1)
except:
    df2 = pd.DataFrame(columns = ['type', 'r', 'lr', 'epoch', 'loss_val', 'rocauc_val', 'run'])
    run = 0
    

rs = [1.0]
modes = ['CNNModel', 'CNNModel_transf', 'CNNModel_freezing']
epochs = 35
adam_lr = [1e-05, 5e-05, 0.000075]


if 'CNNModel' in modes:
    for lr in adam_lr:
        for r in rs:
            dx = dx_f(r)
            model = CNNModel(learning_rate=lr)
            early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
            trainer = Trainer(max_epochs = epochs, deterministic=True, callbacks=[early_stop_callback], num_sanity_val_steps=0)
            trainer.fit(model=model, datamodule=dx)
            df = pd.DataFrame(model.story_val, columns = ['epoch', 'loss_val', 'rocauc_val'])
            df['type'] = 'CNNModel'
            df['r'] = r
            df['lr'] = lr
            df['run'] = run
            df = df[['type', 'r', 'lr', 'epoch', 'loss_val', 'rocauc_val', 'run']]
            df2 = df2.append(df).reset_index(drop=True)
            df2.to_csv('../Results/transf_metr_m4.tsv', sep='\t', index=None)

if 'CNNModel_transf' in modes:
    for lr in adam_lr:
        for r in rs:
            dx = dx_f(r)
            model = CNNModel(learning_rate=lr)
            model.load_state_dict(torch.load('../Results/models/Abs_m4_e19.pt'))
            model.eval()
            early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
            trainer = Trainer(max_epochs = epochs, deterministic=True, callbacks=[early_stop_callback], num_sanity_val_steps=0)
            trainer.fit(model=model, datamodule=dx)
            df = pd.DataFrame(model.story_val, columns = ['epoch', 'loss_val', 'rocauc_val'])
            df['type'] = 'CNNModel_transf'
            df['r'] = r
            df['lr'] = lr
            df['run'] = run
            df = df[['type', 'r', 'lr', 'epoch', 'loss_val', 'rocauc_val', 'run']]
            df2 = df2.append(df).reset_index(drop=True)
            df2.to_csv('../Results/transf_metr_m4.tsv', sep='\t', index=None)
            
if 'CNNModel_freezing' in modes:
    for lr in adam_lr:
        for r in rs:
            dx = dx_f(r)
            model = CNNModel(learning_rate=lr)
            model.load_state_dict(torch.load('../Results/models/Abs_m4_e19.pt'))
            model.eval()
            for param in list(model.parameters())[:2]:
                param.requires_grad = False
            early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
            trainer = Trainer(max_epochs = epochs, deterministic=True, callbacks=[early_stop_callback], num_sanity_val_steps=0)
            trainer.fit(model=model, datamodule=dx)
            df = pd.DataFrame(model.story_val, columns = ['epoch', 'loss_val', 'rocauc_val'])
            df['type'] = 'CNNModel_freezing'
            df['r'] = r
            df['lr'] = lr
            df['run'] = run
            df = df[['type', 'r', 'lr', 'epoch', 'loss_val', 'rocauc_val', 'run']]
            df2 = df2.append(df).reset_index(drop=True)
            df2.to_csv('../Results/transf_metr_m4.tsv', sep='\t', index=None)
            
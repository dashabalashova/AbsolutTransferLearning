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


torch.set_num_threads(5)
os.makedirs('../Results/', exist_ok = True)

r = 1
k = 0
random_seed = 0

seed_everything(random_seed, workers=True)
mod = CNNModel(learning_rate=0.000075)
dx = AbLightDataset(k = k, r = r, batch_size = 16, random_seed = random_seed, data_f = 'data_Absolut4.tsv')
dx.setup()

os.makedirs('../Code/models/', exist_ok = True)
roc_aucs = []
for n in range(20):
    trainer = Trainer(max_epochs = 1, deterministic=True, check_val_every_n_epoch=1)
    trainer.fit(model=mod, datamodule=dx)
    roc_auc, lloss = f1(mod, dx, trainer)
    print('Epoch: %s, ROC_AUC: %s' %(n, roc_auc))
    roc_aucs.append([n, roc_auc])
    torch.save(mod.state_dict(), '../Results/models/Abs_m4_e'+str(n)+'.pt')
df = pd.DataFrame(roc_aucs, columns = ['epoch', 'roc_auc'])
df.to_csv('../Results/roc_auc_m4.tsv', sep='\t', index=None)

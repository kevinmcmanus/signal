import numpy as np
import pandas as pd

import os

datapath = '~/TESS/s0026'

#
toi_catalog = pd.read_csv(os.path.join(datapath,'csv-file-toi-catalog.csv'), skiprows=4, index_col='TIC')

toi_catalog = pd.read_csv('https://tev.mit.edu/toi/toi-release', skiprows=4, index_col='TIC')
toi_catalog.index.rename('TIC_ID', inplace=True)

target_s26 = pd.read_csv('https://archive.stsci.edu/hlsps/qlp/target_lists/s0026.csv')
target_s26.rename({'#TIC_ID':'TIC_ID'}, axis='columns', inplace=True)
target_s26.set_index('TIC_ID',inplace=True)

s26_toi = toi_catalog.join(target_s26, how='inner')
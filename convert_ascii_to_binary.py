import pandas as pd
import numpy as np
import pickle

def convert_tab_delimited_ascii_to_binary_and_memmap(target_file):
    
    print('Loading CSV...')
    data = pd.read_csv(target_file, delimiter='\t', names=['channel1', 'channel2']).values
    print(f'... CSV loaded. Its dtype: {data.dtype}, its shape: {data.shape}. Converting to float32...')
    data = np.float32(data)
    print('Saving to binary NPY file...')
    np.save(target_file + '.npy', data, allow_pickle=False)
    print('...done.')

convert_tab_delimited_ascii_to_binary_and_memmap(r"13V_2pt5h_50ms") #insert you file location



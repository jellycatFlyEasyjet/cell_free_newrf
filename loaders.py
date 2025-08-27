#%%
import numpy as np
import pandas as pd
from typing import List
from omegaconf import OmegaConf
from config_function import ap_id
class DatasetLoader:
    def __init__(self, dataset_file) -> None:
        self.dataset = pd.read_pickle(dataset_file)
        self.dataset = self.dataset.dropna()
        self.trainset = []
        self.valset = []
        self.testset = []

    def get_pair(self, tx_id, rx_id):
        pair = self.dataset[(self.dataset['TxID'] == tx_id) & (self.dataset['RxID'] == rx_id)]
        return pair

    def get_cfr(self, tx_id, rx_id):
        pair = self.get_pair(tx_id, rx_id)
        return pair['CSI'].to_numpy()[0].astype(np.complex64)
    
    def get_cfr_batch(self, tx_ids, rx_ids):
        cfr_lst = []
        for rx_id in rx_ids:
            if len(tx_ids) == 1:
                cfr_lst.append(self.get_cfr(tx_ids[0], rx_id).flatten())
            else:
                for tx_id in tx_ids:
                    cfr_lst.append(self.get_cfr(tx_id, rx_id).flatten())
        return np.array(cfr_lst)
    
    def get_freq(self):
        return self.dataset['Frequency'].unique()[0].astype(np.float32) 
    
    def get_loc(self, dev_type, id):
        if dev_type == "AP":
            dev_entries = self.dataset[self.dataset['RxID'] == id]['TxPos']
            dev_entries = dev_entries.tolist()
            return dev_entries
        elif dev_type == "STA":
            dev_entries = self.dataset[self.dataset['RxID'] == id]['RxPos']
            return dev_entries[dev_entries.index[0]]
    
    def get_loc_batch(self, dev_type, ids):
        if dev_type == "AP":
            AP_loc = []
            dev_entries = self.dataset[self.dataset['RxID'] == ids[0]]['TxPos']
            for i in dev_entries.index:
                AP_loc.append(dev_entries[i])
            return np.array(AP_loc).astype(np.float32)
        elif dev_type == "STA":
            loc_lst = []
            for i in ids:
                loc_lst.append(self.get_loc(dev_type, i))
            return np.array(loc_lst).astype(np.float32)
    
    def get_aoa(self, id):
        lastXpts,raysNum = self.get_last_itx_pts(id)
        center = self.get_loc('STA', id)            
        aoa = lastXpts-center
        return aoa/np.linalg.norm(aoa,axis=1)[:, None].astype(np.float32),raysNum
    
    def get_aoa_batch(self, ids) -> List[np.ndarray]:
        aoa_lst = []
        aoaNum_lst = []
        for i in ids:
            lastXpts,raysNum = self.get_aoa(i)
            aoa_lst.append(lastXpts)
            aoaNum_lst.append(raysNum)
        return aoa_lst,aoaNum_lst
    
    def get_last_itx_pts(self, id):
        if len(ap_id) == 1:           
            dev_entries = self.dataset[(self.dataset['TxID'] == ap_id[0]) & (self.dataset['RxID'] == id)]['LastXPts']
            return dev_entries[dev_entries.index[0]].T, dev_entries[dev_entries.index[0]].T.shape[0] # 返回最后一次的LastXPts和空的raysNum列表

        else:
            dev_entries = self.dataset[self.dataset['RxID'] == id]['LastXPts']
            
            if dev_entries.empty:
                return None, []  # 或者 raise Exception("No entries found for RxID")

            lastPts_list = []
            rays_list = []

            for entry in dev_entries:
                lastPts = entry.T
                lastPts_list.append(lastPts)
                rays_list.append(lastPts.shape[0])

            all_lastPts = np.concatenate(lastPts_list, axis=0)
            return all_lastPts, rays_list
        
    def get_last_itx_pts_batch(self, id_list):
        lastXPts_all = []
        raysNum_list = []

        for rx_id in id_list:
            rx_id = int(rx_id)  # 确保是 int 类型

            if len(ap_id) == 1:
                dev_entries = self.dataset[
                    (self.dataset['TxID'] == ap_id[0]) & (self.dataset['RxID'] == rx_id)
                ]['LastXPts']
                
                if dev_entries.empty:
                    lastXPts_all.append(None)
                    raysNum_list.append(0)
                    continue

                lastPts = dev_entries.iloc[0].T
                lastXPts_all.append(lastPts)
                raysNum_list.append(lastPts.shape[0])
            else:
                dev_entries = self.dataset[self.dataset['RxID'] == rx_id]['LastXPts']
                
                if dev_entries.empty:
                    lastXPts_all.append(None)
                    raysNum_list.append(0)
                    continue

                lastPts_list = [entry.T for entry in dev_entries]
                rays_list = [pts.shape[0] for pts in lastPts_list]

                all_lastPts = np.concatenate(lastPts_list, axis=0)
                lastXPts_all.append(all_lastPts)
                raysNum_list.append(sum(rays_list))

        return lastXPts_all, raysNum_list

    def get_station_ids(self):
        return self.dataset['RxID'].unique()
    
    def get_ap_ids(self):
        return self.dataset['TxID'].unique()

    def split_train_val_test(self, train_ratio=0.8, val_ratio=0.1, seed=12):
        '''
        Split the dataset into training, validation, and test sets.
        @param train_ratio: The ratio of the training set to the whole dataset.
        @param val_ratio: The ratio of the validation set to the whole dataset.
        @param seed: Random seed for reproducibility.
        Note: test_ratio = 1 - train_ratio - val_ratio
        '''
        all_rx = self.get_station_ids()
        np.random.seed(seed)  # for reproducibility
        shuffled_rx = np.random.permutation(all_rx)

        n_train = int(len(shuffled_rx) * train_ratio)
        n_val = int(len(shuffled_rx) * val_ratio)
        
        self.trainset = shuffled_rx[:n_train]
        self.valset = shuffled_rx[n_train:n_train+n_val]
        self.testset = shuffled_rx[n_train+n_val:]
    
    def split_train_val(self, ratio = 0.8, seed=12):
        '''
        Split the dataset into training and validation sets.
        @param ratio: The ratio of the training set to the whole dataset.
        @param fixed_val: If True, the validation set will be fixed to the last 20% of the dataset.
        '''
        all_rx = self.get_station_ids()
        np.random.seed(seed)  # for reproducibility
        shuffled_rx = np.random.permutation(all_rx)

        n_train = int(len(shuffled_rx) * ratio)
        self.trainset = shuffled_rx[:n_train]
        self.valset = shuffled_rx[n_train:]
#%%
if __name__ == "__main__":
    loader = DatasetLoader("./data/dataset_conference_ch1_rt_image_fc.pkl")
    loader.split_train_val()
    trainset = loader.trainset
    valset = loader.valset
    freq = loader.get_freq()
    loc_sta_1 = loader.get_loc('STA', 1)
    loc_batch_sta = loader.get_loc_batch('STA', [1, 2, 3])
    aoa_1 = loader.get_aoa(1)
    aoa_batch = loader.get_aoa_batch([1, 2, 3])
    last_itx_pts_1 = loader.get_last_itx_pts(1)
    station_ids = loader.get_station_ids()
    ap_ids = loader.get_ap_ids()
    cfr_1_1 = loader.get_cfr(1, 1)
    cfr_batch_1 = loader.get_cfr_batch(1, [1, 2, 3])

    print(f"trainset size: {trainset.shape}, type: {type(trainset)}")
    print(f"valset size: {valset.shape}, type: {type(valset)}")
    print(f"freq size: {freq.shape}, type: {type(freq)}, freq value: {freq}")
    print(f"loc_sta_1 size: {loc_sta_1.shape}, type: {type(loc_sta_1)}")
    print(f"loc_batch_sta size: {loc_batch_sta.shape}, type: {type(loc_batch_sta)}")
    print(f"aoa_1 size: {aoa_1.shape}, type: {type(aoa_1)}")
    print(f"aoa_batch size: {len(aoa_batch)}, type: {type(aoa_batch)}, aoa_batch[0] size: {aoa_batch[0].shape}")
    print(f"last_itx_pts_1 size: {last_itx_pts_1.shape}, type: {type(last_itx_pts_1)}")
    print(f"station_ids size: {station_ids.shape}, type: {type(station_ids)}")
    print(f"ap_ids size: {ap_ids.shape}, type: {type(ap_ids)}")
    print(f"cfr_1_1 size: {cfr_1_1.shape}, type: {type(cfr_1_1)}, dtype: {cfr_1_1.dtype}")
    print(f"cfr_batch_1 size: {cfr_batch_1.shape}, type: {type(cfr_batch_1)}, dtype: {cfr_batch_1.dtype}")
# %%

import os.path as osp
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden
import pickle
import random
from scipy.io import loadmat
import os
from dassl.utils import read_json, write_json, mkdir_if_missing
from .oxford_pets import OxfordPets
def read_dataset(file_path):
        data = {}
        with open(file_path, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                label = int(label)
                if label not in data:
                    data[label] = []
                data[label].append(path)
        return data
@DATASET_REGISTRY.register()
class DOMAINNETNEW(DatasetBase):
    """DomainNet.
    """

    dataset_dir = "/raid/biplab/hassan/PromptSRC/data/DomainNet-126"
    domains = ["infograph","clipart","painting","real","quickdraw","sketch"]

    def __init__(self, cfg):
        print(cfg.DATASET.SOURCE_DOMAINS)
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        
        
        self.target_dir = osp.join(self.dataset_dir, 'splits')
        self.image_target_dir = osp.join(self.dataset_dir, cfg.DATASET.TARGET_DOMAINS[0])
        # self.classnames = listdir_nohidden(self.image_target_dir )
        # print(self._classnames)
        # self.classnames = [c for c in self.classnames]
        # self.classnames.sort()
        # self.split_dir = osp.join(self.dataset_dir, "splits")
        self.split_target_path = os.path.join(self.dataset_dir, f"split_domainnet_{cfg.DATASET.TARGET_DOMAINS[0]}.json")
        self.split_target_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot_{cfg.DATASET.TARGET_DOMAINS[0]}")
        
        mkdir_if_missing(self.split_target_fewshot_dir)

        mkdir_if_missing(self.split_target_fewshot_dir)

        if os.path.exists(self.split_target_path):
            _,_, testx = OxfordPets.read_split(self.split_target_path, self.image_target_dir)
        else:
            train_split = osp.join(self.target_dir,cfg.DATASET.TARGET_DOMAINS[0]+"_train.txt")
            test_split = osp.join(self.target_dir,cfg.DATASET.TARGET_DOMAINS[0]+"_test.txt")
            train, val, testx = self.splits(self.dataset_dir,train_split,test_split)
            OxfordPets.save_split(train, val, testx, self.split_target_path, self.image_target_dir)
        
        train_set =[]
        val_set =[]
        source_domains = cfg.DATASET.SOURCE_DOMAINS[0].split(",") 
        strs =""
        for i in range(len(source_domains)-1):
            strs+=source_domains[i]+"_"
        preprocessed_set = strs +source_domains[-1]

        for i in range(len(source_domains)):
            self.dataset_dir = osp.join(root, self.dataset_dir)
            self.image_source_dir = osp.join(self.dataset_dir, source_domains[i])
            # self.split_dir = osp.join(self.dataset_dir, "splits")
            self.split_source_path = os.path.join(self.dataset_dir, f"split_domainnnet_{source_domains[i]}.json")
            self.split_source_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot_{source_domains[i]}")
            print(self.image_source_dir)
            mkdir_if_missing(self.split_source_fewshot_dir)
            # print(self.image_target_dir)
            if os.path.exists(self.split_source_path):
                train, val, test = OxfordPets.read_split(self.split_source_path, self.image_source_dir)
            else:
                train_split = osp.join(self.target_dir,source_domains[i]+"_train.txt")
                test_split = osp.join(self.target_dir,source_domains[i]+"_test.txt")
                train, val, test = self.splits(self.dataset_dir,train_split,test_split)
                OxfordPets.save_split(train, val, test, self.split_source_path, self.image_source_dir)
            train_set.extend(train)
            val_set.extend(val)
        
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            few_shot_dir = os.path.join(self.dataset_dir,preprocessed_set)
            preprocessed = os.path.join(few_shot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            mkdir_if_missing(few_shot_dir)
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train_set, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val_set, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        super().__init__(train_x=train, val=val, test=testx)

    @staticmethod
    def splits(dataset_dir,train_path,test_path,p_val=0.3):
        categories = listdir_nohidden(dataset_dir+"/clipart")
        categories = [c for c in categories]
        categories.sort()
        train, val,test = [], [],[]
        data = read_dataset(train_path)
        def _collate(ims, y):
            items = []
            for im in ims:
                c_name = im.split("/")[1]
                im = osp.join(dataset_dir,im)
                item = Datum(impath=im, label=y, classname=c_name)  # is already 0-based
                items.append(item)
            return items
        for c in data.keys():
            n_total = len(data[c])
            n_val = round(n_total * p_val)
            n_train = n_total - n_val
            train.extend(_collate(data[c][:n_train], c))
            val.extend(_collate(data[c][n_train : n_train + n_val], c))
        data_test = read_dataset(test_path)

        for c in data_test.keys():
            test.extend(_collate(data_test[c], c))
    
        return train, val, test


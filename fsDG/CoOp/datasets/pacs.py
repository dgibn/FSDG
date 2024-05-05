import os.path as osp
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden
import pickle
import random
from scipy.io import loadmat
import os
from dassl.utils import read_json, write_json, mkdir_if_missing
from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


@DATASET_REGISTRY.register()
class PACSNEW(DatasetBase):
    """PACS.

    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon
        (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse,
        house and person.

    Reference:
        - Li et al. Deeper, broader and artier domain generalization.
        ICCV 2017.
    """

    dataset_dir = "/home/hassan/pacs_data"
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    
    # the following images contain errors and should be ignored
    # _error_paths = ["sketch/dog/n02103406_4068-1.png"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_source_dir = osp.join(self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS)
        # self.split_dir = osp.join(self.dataset_dir, "splits")
        self.split_source_path = os.path.join(self.dataset_dir, f"split_pacs_{cfg.DATASET.SOURCE_DOMAINS}.json")
        self.split_source_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot_{cfg.DATASET.SOURCE_DOMAINS}")

        self.image_target_dir = osp.join(self.dataset_dir, cfg.DATASET.TARGET_DOMAINS)
        # self.split_dir = osp.join(self.dataset_dir, "splits")
        self.split_target_path = os.path.join(self.dataset_dir, f"split_pacs_{cfg.DATASET.TARGET_DOMAINS}.json")
        self.split_target_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot_{cfg.DATASET.TARGET_DOMAINS}")
        mkdir_if_missing(self.split_source_fewshot_dir)
        mkdir_if_missing(self.split_target_fewshot_dir)

        if os.path.exists(self.split_source_path):
            train, val, test = OxfordPets.read_split(self.split_source_path, self.image_source_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_source_dir)
            OxfordPets.save_split(train, val, test, self.split_source_path, self.image_source_dir)
        
        if os.path.exists(self.split_target_path):
            _,_, testx = OxfordPets.read_split(self.split_target_path, self.image_target_dir)
        else:
            _, _, testx = DTD.read_and_split_data(self.image_target_dir)
            OxfordPets.save_split(train, val, testx, self.split_target_path, self.image_target_dir)
        
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_source_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        # print(train)
        # print(textx)
        super().__init__(train_x=train, val=val, test=testx)
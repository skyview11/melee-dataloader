# dataloader

### convert slp to memmap
``` shell
python3 data_preprocess.py --source_path your/data/path --target_path memmap/final/path
```

### use dataloader
First, change configuration_dataset.py fit to your setting, 
or initailize. 

``` python
from slippi_dataset import SlippiDataset
from configuration_dataset import DatasetConfig

config_dict = {
    "dataset_basepath": "/root/data/skyview/mm", 
    "seq_len": 1
}
config = DatasetConfig(config_dict)

train_dataset = SlippiDataset(config)
...
```

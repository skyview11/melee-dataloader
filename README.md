# dataloader

### convert slp to memmap
``` shell
python3 data_preprocess.py --source_path your/data/path --target_path memmap/final/path
```

### use dataloader
``` python
from slippi_dataset import SlippiDataset
...
```
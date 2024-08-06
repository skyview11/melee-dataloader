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

### Dataloader information

데이터로더는 위 코드의 dataset_basepath로 정의된 경로 아래 있는 모든 memmmap 객체 (*.dat) 를 읽는다. 현재 `/root/data/skyview/mm` 경로에 약 1600 개의 리플레이파일을 memmap으로 추출한 결과물이 있기 때문에 해당 폴더를 `dataset_basepath` 아래 놓으면 코드가 정상적으로 동작한다. 

`seq_len` 은 연속된 몇 step의 데이터를 불러올지를 결정한다. 예를 들어 `seq_len` 을 10 으로 정의하고, dataset의 100 번 index를 요청한 경우, 100번 인덱스부터 107번 인덱스까지를 반환한다. 만약 게임이 101번 인덱스에서 끝날 경우, 남은 time step는 zero-padding 된다. 

dataset의 길이는 모든 리플레이파일의 time step 개수의 합이다. 

dataset의 `__getitem__` 메써드는 모든 리플레이를 time step 기준으로 연속적으로 쌓았을 때의 입력한 인덱스에 대응하는 `p1_state`, `p1_action`, `p1_reward`, `p2_state`, `p2_action`, `p2_reward` 를 딕셔너리로 반환한다. 


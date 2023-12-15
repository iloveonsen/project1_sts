## Demo Grid Search Model

밤새 모델을 돌려놓을 목적으로 baseline 을 refactoring 한 코드 입니다.

`config` 폴더 내부에 `config.json` 파일에서 hyper-parameter 를 수정하신 뒤

```json
{
    "seed": 1784,
    "inference": false,
    "wandb_project_name": "<your-wandb-project-name>",
    "wandb_username": "<your-wandb-user-name>",
    "model_name": ["klue/roberta-base", "klue/roberta-small"],
    "model_detail": ["v", "v"],
    "resume": false,
    "batch_size": [16, 32, 64],
    "max_epoch": [5, 10, 15],
    "shuffle": true,
    "learning_rate": [1e-5, 5e-5],
    "kfold": 5,
    "data_dir": "./data",
    "output_dir": "./outputs",
    "model_dir": "./saves",
    "train_path": "train.csv",
    "dev_path": "dev.csv",
    "test_path": "dev.csv",
    "predict_path": "test.csv"
}
```
- `data` 폴더 내부에 `train.csv`, `dev.csv`, `test.csv`, `sample_submission.csv` 를 넣어놔주세요.
- `model_name` 의 list 에 원하는 모델을 추가가능합니다.
- `model_detail` 의 경우 `model_name` 와 같은 인덱스에 대응해야 합니다.
- `batch_size`, `max_epoch`, `learning_rate` 에 명시된 각 원소들을 조합하여 모든 combination 에 대해 모델을 실행합니다.
- `kfold` 값을 1보다 크게 설정한 경우 cross-validation 으로 학습합니다.
  - 이때 각 fold 당 학습횟수는 `max_epoch // kfold` 입니다.


`python run.py` 로 training, `python run.py --inference=true` 로 inference 해주시면 실행됩니다.

모델 저장은 `./saves/` 내부에 저장됩니다.
- 저장 path 는 `./saves/klue/roberta-small_v03_16_1_1e-05_000_00583_0.862_20231214_221830.ckpt`형식 이며
- 이름 `roberta-small_v03_16_1_1e-05_000_00583_0.862_20231214_221830` 은
  - `roberta-small`: 모델명
  - `v03`: 버전
    - 버전은 매 combination 마다 자동으로 update 됩니다.
  - `16`: batch_size
  - `1`: max_epoch
  - `1e-05`: learning_rate
  - `000`: current_epoch
  - `00583`: current_step
  - `0.862`: pearson value
  - `20231214_221830`: current_date _ current_time

inference 하실경우, `./outputs/` 에 저장됩니다.


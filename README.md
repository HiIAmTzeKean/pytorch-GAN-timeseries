# Financial time series generation using GANs

## Dataset

The dataset used in this project is a daily stock time series dataset, automatically downloaded from Yahoo Finance using the `yfinance` library. The main class for handling the data is `StockDataset` (see `dataset.py`).

- **Data source:** The dataset is constructed by downloading the daily closing prices for a specified ticker symbol (e.g., 'D05.SI' for DBS Group) over a user-defined date range. The data is retrieved as a sequence of daily closing prices, grouped by date.
- **Format:** Each sample in the dataset is a sequence (by default, one per day) of closing prices, stored as a PyTorch tensor of shape `(sequence_length, 1)`.
- **Normalization:** By default, the data is normalized to the range `[-1, 1]` for stable GAN training. The normalization parameters are stored to allow for denormalization later.
- **Delta statistics:** The dataset computes the difference (delta) between the last and first value of each sequence, and fits a Gaussian distribution to these deltas. This allows for conditional generation: the generator can be conditioned on a sampled delta, and the dataset provides methods to sample deltas from the fitted distribution and to normalize/denormalize deltas.
- **Robustness:** If no data is found for the specified ticker or date range, the dataset raises an error to alert the user.

This design allows for flexible, robust, and reproducible experiments on real-world financial time series, with support for both unconditional and delta-conditioned GAN training.

## Project structure

The files and directories composing the project are:
- `main.py`: runs the training. It can save the model checkpoints and images of generated time series, and features visualizations (loss, gradients) via tensorboard. Run `python main.py -h` to see all the options.
- `generate_dataset.py`: generates a fake dataset using a trained generator. The path of the generator checkpoint and of the output \*.npy file for the dataset must be passed as options. Optionally, the path of a file containing daily deltas (one per line) for conditioning the time series generation can be provided.
- `finetune_model.py`: uses pure supervised training for finetuning a trained generator. *Discouraged*, it is generally better to train in supervised and unsupervised way jointly. 
- `models/`: directory containing the model architecture for both discriminator and generator.
- `utils.py`: contains some utility functions. It also contains a `DatasetGenerator` class that is used for fake dataset generation.
- `main_cgan.py`: runs training with standard conditional GANs. Cannot produce nice results, but it is kept for reference.

By default, during training, model weights are saved into the `checkpoints/` directory, snapshots of generated series into `images/` and tensorboard logs into `log/`.

Use:
```
tensorboard --logdir log
```
from inside the project directory to run tensoboard on the default port (6006).

## Examples
Run training with recurrent generator and convolutional discriminator, conditioning generator on deltas and alternating adversarial and supervised optimization:

```shell
python main.py --delta_condition --gen_type lstm --dis_type cnn --alternate --run_tag cnn_dis_lstm_gen_alternte_my_first_trial
```

Generate fake dataset `prova.npy` using deltas contained in `delta_trial.txt` and model trained for 70 epochs:

```shell
python generate_dataset.py --delta_path delta_trial.txt --checkpoint_path checkpoints/cnn_conditioned_alternate1_netG_epoch_70.pth --output_path prova.npy
```

Finetune checkpoint of generator with supervised training:

```shell
python finetune_model.py --checkpoint checkpoints/cnn_dis_lstm_gen_noalt_new_netG_epoch_39.pth --output_path finetuned.pth
```

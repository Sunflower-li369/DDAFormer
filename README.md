# DDAFormer

# [DDAFomrer: Dual Differential Attention Transformer for Medical Image Segmentation]

## Updates

- 2025.7


### Datasets
Download the Synapse dataset from [here](https://drive.google.com/uc?export=download&id=18I9JHH_i0uuEDg-N6d7bfMdf7Ut6bhBi).


### Training and Testing

1) Install the Requirements:

    `pip install -r requirements.txt`

2) Train:
    ```bash
    python train.py --batch_size 20 --max_epochs 440 --module networks.DDAFormer.DDAFormer --eval_interval 20 
    ```

 3) Test:
    ```bash
    python test.py --batch_size 20 --max_epochs 440 --module networks.DDAFormer.DDAFormer --eval_interval 20 
    ```

## Experiment Results

# Enhancing Images with Uneven Illumination using Ensemble Learning (EIUIE)
This repository contains the codebase for the EIUIE approach to enhancing images with uneven illumination.

## Usage
To enhance a single image, the following cammand can be used:

```bash
python eiuie/main.py single --file=<PATH_TO_FILE> --method=unsharp_masking
python eiuie/main.py single --file=<PATH_TO_FILE> --method=retinex
python eiuie/main.py single --file=<PATH_TO_FILE> --method=homomorphic_filtering
python eiuie/main.py single --file=<PATH_TO_FILE> --method=fusion
```

In order to use the fusion model, its parameters have to be trained first.

## Training
First, the [LOL-dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset) has to be downloaded, extracted, and moved to `data/lol_dataset/`.

From this dataset training samples will be generated. This happens in two steps:
In the first step, the three base methods (unsharp_masking, retinex, homomorphic_filtering) are applied to all images in the LOL-dataset. Results will be stored in `data/intermediate_images/`
```bash
python eiuie/main.py batch_process
```
Afterwards, the training dataset consisting of all pixel samples can be generated to `data/pixel_dataset/` using:
```bash
python eiuie/main.py prepare_dataset
```

Training can then be conducted, with the best model parameters (`best_model.pt`) as well as further checkpoints being saved to `data/checkpoints/`, using:
```bash
python eiuie/main.py train
```

The learnt parameters can be pretty printed using:
```bash
python eiuie/main.py ppparams
```

Finally, training can be evaluated on a custon evaluation dataset (`data/eval_dataset/input/`) with the following command. Output images and scores for different evaluation metrics can be found afterwards in `data/eval_dataset`.
```bash
python eiuie/main.py eval
```


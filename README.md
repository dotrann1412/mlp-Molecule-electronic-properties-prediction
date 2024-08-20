# Molecule electronic properties prediction using MLP

The method is based on: [Atom-Based Machine Learning Model for Quantitative Property–Structure Relationship of Electronic Properties of Fusenes and Substituted Fusenes](https://pubs.acs.org/doi/10.1021/acsomega.3c05212).

## Train

```sh
python train.py -d data/raw_cyano_data.csv data/raw_nitro_data.csv data/raw_pah_data.csv \
    -e 10 \ # epochs
    -f smiles \ # use smiles string as model input
    -t IP EA BG \ # targets for making predictions
    -hl 256 256 # hidden layers of the mlp model
```

Best loss: ~$0.1$

## Validation

```sh
python eval.py -cfg ./artifacts/model_store.json -w artifacts/model.pth
```
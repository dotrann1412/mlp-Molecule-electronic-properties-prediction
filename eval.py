import torch
from argparse import ArgumentParser
import json
from model import Regressor
import os
import pandas as pd
import numpy as np
from magic.vectorization import GraphVectorizer
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser('Train a regressor model on a dataset')

    # required arguments
    parser.add_argument('-cfg', '--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('-w', '--weight', type=str, required=True, help='Path to the model weight')

    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Regressor(
        input_size = config['meta']['features'], 
        output_size = config['meta']['targets'], 
        hidden_layers = config['hidden_layers']
    )

    _mx, _mn = np.array(config['meta']['data']['max']), np.array(config['meta']['data']['min'])
    vectorizer = GraphVectorizer.from_json(config['meta']['vectorizer'])
    
    _mx, _mn = torch.from_numpy(_mx).to(device), torch.from_numpy(_mn).to(device)
    
    model.load_state_dict(torch.load(args.weight, map_location=device), strict=True)
    
    model.eval()
    model.to(device)
    
    files = [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.csv')]
    
    df = pd.concat([pd.read_csv(f) for f in files])
    df.dropna(inplace=True)
    predictions = []
    
    with torch.no_grad():
        for i, smile in tqdm(enumerate(df['smiles'].values), total=len(df)):
            x = vectorizer.transform([smile])
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = model(x).squeeze(0)
            ip, ea, bg = y * (_mx - _mn) + _mn
            
            
            predictions.append((smile, ip.item(), df['IP'].values[i], ea.item(), df['EA'].values[i], bg.item(), df['BG'].values[i]))

    predictions = pd.DataFrame(predictions, columns=['smiles', 'ip', 'ip_true', 'ea', 'ea_true', 'bg', 'bg_true'])
    predictions.to_csv('predictions.csv', index=False)
    
if __name__ == '__main__':
    main()
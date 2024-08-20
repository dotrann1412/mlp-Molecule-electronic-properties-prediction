from argparse import ArgumentParser
import pandas as pd
import numpy as np
import torch
from magic.vectorization import GraphVectorizer
from model import Regressor
import json
import os
from data import MoleculeDataset

class ModifiedRMSE(torch.nn.Module):
    def __init__(self, _max, _min, ratio=0.6):
        super(ModifiedRMSE, self).__init__()
        self.max = torch.from_numpy(_max)
        self.min = torch.from_numpy(_min)
        self.ratio = ratio
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, y):
        # unnorm_pred = pred * (self.max - self.min) + self.min
        # unnorm_y = y * (self.max - self.min) + self.min
        x = torch.sqrt(self.mse(pred, y)) # + (1 - self.ratio) * torch.sqrt(((unnorm_y[:,2] + unnorm_pred[:,1] - unnorm_pred[:,0]) ** 2).mean())
        return x

def parse_args():
    parser = ArgumentParser('Train a regressor model on a dataset')

    # required arguments
    parser.add_argument('-d', '--data', type=str, required=True, help='Path to the dataset files (csv)', nargs='+')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('-o', '--output', type=str, default='artifacts', help='Path to save the model')

    # specify in features and targets 
    parser.add_argument('-f', '--feature', type=str, required=True, help='List of features', nargs='+')
    parser.add_argument('-t', '--target', type=str, required=True, help='List of targets', nargs='+')
    parser.add_argument('-hl', '--hidden-layers', type=int, required=False, default=[], help='List of dense unit', nargs='+')
    parser.add_argument('-norm', '--normalization', type=str, default='maxmin', choices=['maxmin', 'zscore'], 
                        help='Normalization method for the targets. Default is zscore')

    # optionals
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('-lr', '--init-lr', type=int, default=1e-3, help='Initial learning rate')
    parser.add_argument('-r', '--train-ratio', type=float, default=.7, help='Initial learning rate')
    parser.add_argument('-s', '--random-seed', type=int, default=2024, help='Random seed for reproducibility')

    # graph processing
    parser.add_argument('-wker', type=str, default='WLSubtree', help='WL kernel type. Choice: subtree/wla, edge/wlab, shortest_path/wlad. '
                        'Alternatively, choose ecfp for ECFP fingerprint. Default is subtree/wla', 
                        choices = ["WLSubtree","WLEdge", "WLShortestPath"])
    parser.add_argument('-iter', type=int, default=5, help='Number of iterations for the WL kernel')

    return parser.parse_args()

def preload_dataset(
    inp_files, 
    inp_features, 
    inp_targets, 
    train_ratio=.7, 
    wker='WLSubtree', 
    num_iter=5
):
    df = pd.concat([pd.read_csv(f) for f in inp_files])
    df.dropna(inplace=True)
    
    features = df[inp_features].values.tolist()
    targets = df[inp_targets]

    vectorizer = GraphVectorizer(wker, num_iter=num_iter, smiles=True)
    
    features_vector = vectorizer.fit(features).transform(features)

    train_features, test_features = features_vector[:int(len(features) * train_ratio)], features_vector[int(len(features) * train_ratio):]
    train_targets, test_targets = targets[:int(len(targets) * train_ratio)], targets[int(len(targets) * train_ratio):]

    train_targets = np.array(train_targets)
    test_targets = np.array(test_targets)

    targets_mean, targets_std = train_targets.mean(axis=0), train_targets.std(axis=0)
    targets_min, targets_max = train_targets.min(axis=0), train_targets.max(axis=0)

    return train_features, train_targets, test_features, test_targets, {
        'data': {
            'mean': targets_mean, 
            'std': targets_std,
            'min': targets_min,
            'max': targets_max
        },
        'vectorizer':  vectorizer.to_json(),
    }

def default_json_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    
    return str(obj)

def main():
    opts = parse_args()

    inp_files = [e for e in opts.data if e.endswith('.csv')]
    assert len(opts.feature) > 0

    inp_features = opts.feature[0] 
    inp_targets = opts.target

    # fix the seed
    np.random.seed(opts.random_seed)
    torch.manual_seed(opts.random_seed)

    model_store = {}

    train_features, train_targets, test_features, test_targets, meta = preload_dataset(
        inp_files, inp_features, inp_targets, 
        train_ratio=opts.train_ratio
    )

    if opts.normalization == 'maxmin':
        train_targets = (train_targets - meta['data']['min']) / (meta['data']['max'] - meta['data']['min'])
        test_targets = (test_targets - meta['data']['min']) / (meta['data']['max'] - meta['data']['min'])
    else:
        train_targets = (train_targets - meta['data']['mean']) / meta['data']['std']
        test_targets = (test_targets - meta['data']['mean']) / meta['data']['std']

    print(f"Training on {len(train_features)} samples, testing on {len(test_features)} samples")
    print(f"Features: {inp_features}, Targets: {inp_targets}")

    print("Normalized targets:")

    print('Train\'s properties: Mean =', train_targets.mean(axis=0), '; Std =', train_targets.std(axis=0))
    print('Test\'s properties: Mean =', test_targets.mean(axis=0), '; Std =', test_targets.std(axis=0))

    print("Properties:")
    print(train_features.shape, test_features.shape)

    model_in_features = train_features.shape[1]
    model_out_features = len(inp_targets)

    model = Regressor(
        model_in_features, 
        model_out_features, 
        hidden_layers=opts.hidden_layers
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params} parameters")

    model_store = {
        'hidden_layers': opts.hidden_layers,
        'meta': {
            'features': model_in_features,
            'targets': model_out_features,
            **meta
        },
    }
    
    print(model_store.keys())
    print(model_store['meta']['data'])

    os.makedirs(opts.output, exist_ok=True)
    with open(os.path.join(opts.output, 'model_store.json'), 'w') as f:
        json.dump(model_store, f, default=default_json_serializer)

    train_ds = MoleculeDataset(train_features, transforms=None, labels=train_targets)
    test_ds = MoleculeDataset(test_features, transforms=None, labels=test_targets)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=opts.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=opts.init_lr)
    # criterion = torch.nn.MSELoss()    
    criterion = ModifiedRMSE(meta['data']['max'], meta['data']['min'])

    for epoch in range(opts.epochs):
        model.train()
        train_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{opts.epochs}, Train Loss: {train_loss}")

    model.eval()

    test_loss = np.array([0.0 for _ in range(len(inp_targets))], dtype=np.float32)
    rmsd = np.array([0.0 for _ in range(len(inp_targets))], dtype=np.float32)
    
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            y_pred = model(x)

            test_loss += np.array([criterion(y_pred[:, i], y[:, i]).item() 
                                   for i in range(len(inp_targets))], dtype=np.float32)

            rmsd += np.array([torch.sqrt(criterion(y_pred[:, i], y[:, i])).item() 
                              for i in range(len(inp_targets))], dtype=np.float32)

    test_loss /= len(test_loader)
    rmsd /= len(test_loader)

    for i, target in enumerate(inp_targets):
        print(f"Target: {target}, Test Loss: {test_loss[i]}, RMSD: {rmsd[i]}")

    torch.save(model.state_dict(), os.path.join(opts.output, 'model.pth'))

if __name__ == '__main__':
    main()
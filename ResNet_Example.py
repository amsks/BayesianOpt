import torch
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet, BasicBlock
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from torchvision.models.resnet import ResNet, BasicBlock
from sklearn.metrics import balanced_accuracy_score

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from typing import List
from typing import Optional
from functools import partial
from typing import Tuple
from typing import Union

from tqdm.autonotebook import tqdm

import numpy as np
import pickle 

import sys
import os
import gin

from BayesianOpt import BayesianOpt

# Customize the RESNET to 9 layers and 10 classes
def create_resnet9_model() -> nn.Module:
    model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model


# Lightning Module helps with optimization based on specified Learning Rate for SGD optimizer
class ResNet9(pl.LightningModule):
    def __init__(self, learning_rate=0.005):
        super().__init__()
        self.model = create_resnet9_model()
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    @auto_move_data
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        loss = self.loss(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


# Return the predicted classes and probabilities
def predict(x, model: pl.LightningModule):
    model.freeze()
    probabilities = torch.softmax(model(x), dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities


# Define the Objective function
def objective(  lr=0.1, 
                epochs=1, 
                gpu_count=1, 
                iteration=None, 
                model_dir='./outputs/models/', 
                train_dl=None,
                test_dl = None 
            ):

    save = False
    
    checkpoint = "current_model.pt"

    if train_dl == None:
        print("Training Data-Loader not specified")
        return 
    elif test_dl == None:
        print("Test Data-Loader not specified")
        return

    if iteration is not None:
        save = True 
        checkpoint = model_dir + f"model_iter_{iteration}.pt"


    model = ResNet9(learning_rate=lr)

    trainer = pl.Trainer(
        gpus=gpu_count,
        max_epochs=epochs,
        progress_bar_refresh_rate=20
    )

    trainer.fit(model, train_dl)
    trainer.save_checkpoint(checkpoint)

    inference_model = ResNet9.load_from_checkpoint(
        checkpoint, map_location="cuda")

    true_y, pred_y, prob_y = [], [], []
    for batch in tqdm(iter(test_dl), total=len(test_dl)):
        x, y = batch
        true_y.extend(y)
        preds, probs = predict(x, inference_model)
        pred_y.extend(preds.cpu())
        prob_y.extend(probs.cpu().numpy())

    if save is False:
        os.remove(checkpoint)

    return np.mean(balanced_accuracy_score(true_y, pred_y))


@gin.configurable(blacklist=['output_dir'])
def session(
    budget=10,
    init_samples=2,
    epochs=1,
    init_epochs=1,
    gpu_count=1,
    batch_size=128,
    output_dir="./output",
    length_scale = 1.0,
    nu=2.5,
    alpha=1e-10,
    n_restarts_optimizer=25,
    epsilon=0.01, 
    eps_decay=False
):

    os.makedirs(os.path.join(output_dir, "plots"))
    os.makedirs(os.path.join(output_dir, "models"))
    
    train_data = KMNIST("kmnist", train=True, download=True, transform=ToTensor())
    test_data = KMNIST("kmnist", train=False, download=True, transform=ToTensor())

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8 )
    test_dl = DataLoader(test_data, batch_size=batch_size, num_workers=8)
    

    # sample the domain
    X = np.array([np.random.uniform(0, 1) for _ in range(init_samples)])
    y = np.array([objective(lr =x, 
                            epochs=init_epochs, 
                            gpu_count=gpu_count,
                            train_dl=train_dl,
                            test_dl=test_dl
                            ) for x in X])

    # reshape into rows and cols
    X = X.reshape(len(X), 1)
    y = y.reshape(len(y), 1)

    # Create the Model
    m52 = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=nu)
    model = GaussianProcessRegressor(kernel=m52, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)


    B = BayesianOpt(GP=model, eps=epsilon, plot_dir=output_dir+"/plots/")

    # perform the optimization process
    for i in range(budget):
        # fit the model
        B.model.fit(X, y)

        # select the next point to sample
        X_next = B.optimize_acq(X, y)

        # sample the point
        Y_next = objective( lr=X_next, 
                            epochs=epochs, 
                            gpu_count=gpu_count,
                            model_dir= output_dir+"/models/", 
                            iteration=i+1,
                            train_dl= train_dl,
                            test_dl = test_dl
                            )

        print(f"LR = {X_next} \t Balanced Accuracy = {Y_next*100} %")


        if i > 0:
            B.plot(X, y, X_next, i+1)

        # add the data to the dataset
        X = np.vstack((X, [[X_next]]))
        y = np.vstack((y, [[Y_next]]))

    for_save = {
        'Learning Rates' : X,
        'Balanced Accuracy' : y
    }

    with open(output_dir + "/history.pkl", "wb") as f:
        pickle.dump(for_save, f)

def main (args):

    # load configuration from Gin file
    if args.opt_config_path is not None:
        gin.parse_config_file(args.opt_config_path)

    del args.opt_config_path
    session(**vars(args))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bayesian Optimization on a RESNET-9 Model for Learning Rate")

    parser.add_argument(
        "--budget", type=int, default=10,
        help="Number of iterations for optimization."
    )

    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch size for ResNet-9 Model"
    )
    
    parser.add_argument(
        "--init_samples", type=int, default=2,
        help="Number of Initial samples before fitting the Gaussian Process"
    )

    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Destination for storing plots and history"
    )

    parser.add_argument(
        "--init_epochs", type=int,  default=1,
        help="Number of Epochs for the initial evaluation"
    )

    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of epochs for training with each value of Learning Rate"
    )

    parser.add_argument(
        "--gpu_count", type=int, default=1, 
        help="Number of GPUs to use for training (0 for only CPU usage)"
    )

    parser.add_argument(
        "--opt_config_path", type=str, default=None,
        help="Path to gin config file to load some or all parameters"
    )

    parser.add_argument(
        "--epsilon", type=float, default=0.01,
        help="Exploration Hyperparameter"
    )
    
    parser.add_argument(
        "--eps_decay", type=bool, default=False,
        help="Set true to decay the epsilon parameter"
    )
    
    args = parser.parse_args()
    main(args)

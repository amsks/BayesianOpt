# Bayesian Optimization

This is a basic and ongoing project on Automated Hyperparameter optimization using Bayesian Optimization. 

NOTE: I haven't written it keping optimality in mind, and probably need to restructure it at some point

## Available Scripts

- BayesianOpt.py : The script containing the class implementation of the Bayesian Optimizer 
- ResNet_Example.py : The script for Implementation of ResNet-9 on the KMNIST dataset and tuning its learning rate

## Packages Used

- Pytorch
- Pytorch Lightning
- torchvision
- typing 
- Numpy
- Matplotlib
- Sci-kit Learn
- Scipy
- Pickle 
- gin 
- tqdm


## Usage

The script can either be run b individually specifying the parameters, which can be found using the ```--help``` command

```
python ResNet_Example.py --help
```

or can be specified in a gin file using the ```opt_config_file``` flag. in the configs folder there are already example gin files, one for Testing with a single epoch and two others for a major run with 50 epochs. A sample comamnd to run this script is given below

```
python ResNet_Example.py --output_dir=./outputs --opt_config_file=./configs/Test_Config.gin
```

## Output

The output directory has a folder to store the models in the iterations and another folder to store the plots of all observations, the posterior mean, uncertainty estimate, and the acquisition function after each iteration, second iteration onwards. 

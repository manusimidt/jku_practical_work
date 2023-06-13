# jku_practical_work


| file                          | description                                                                            |
|-------------------------------|----------------------------------------------------------------------------------------|
| augmentations.py              | contains all augmentation functions, such as rotate, translate, noise..                |
| env.py                        | contains different wrappers for the jumping-task env                                   |
| train_bc.py                   | Training procedure for Behavioural Cloning                                             |
| train_drac.py                 | Training procedure for Data-regularized Actor Critic                                   |
| train_ppo.py                  | Training procedure for PPO                                                             |
| visualize_augmentations.ipynb | Notebook that visualizes the different augmentations on the jumping-task env           |
| visualize_configuration.ipynb | Notebook that visualizes the performance of a trained agent on a certain configuration |
| visualize_env.ipynb           | Notebook that visualizes the jumping-task environment in different configurations      |




``` bash
    conda env create -f environment.yml
    conda activate drl-jumping
```


## Run PPO, RAD and DrAC experiments
A single file `train.py` is used for training PPO, RAD, DrAC and UCB-DrAC. It is possible to 
specify via parameters which algorithm you want to train on which configuration with 
which hyperparameters. The following is an example to train DrAC on the narrow grid
for 2000 episodes using the default hyperparameters

```bash
python train.py -a DRAC -e random -c wide_grid -ne 2000
```

In order to see a list of all parameters, possible values and a short description
use the help command

```bash
python train.py --help
```
which results in the following output:
```
optional arguments:
  -h, --help            show this help message and exit
  -e {vanilla,random,UCB}, --environment {vanilla,random,UCB}
                        The environment to train on
  -c {narrow_grid,wide_grid}, --configuration {narrow_grid,wide_grid}
                        The environment configuration to train on
  -a {PPO,DRAC}, --algorithm {PPO,DRAC}
                        The algorithm to train with
  -ne EPISODES, --episodes EPISODES
                        Number of episodes to train
  -hs HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        Hidden size of the actor and critic network
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for the optimizer
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Size of the minibatch used for one gradient update
  -ap ALPHA_POLICY, --alpha_policy ALPHA_POLICY
                        DrAC alpha weight for policy regularization loss
  -av ALPHA_VALUE, --alpha_value ALPHA_VALUE
                        DrAC alpha weight for value regularization loss
```
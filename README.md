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

# these are needed for pyvirtualdisplay (to create a video of the episode)
``` bash
apt install -y xvfb x11-utils ffmpeg
```
# jku_practical_work

``` bash
    conda env create -f environment.yml
    conda activate drl-jumping
    mkdir gym_jumping_task
    mkdir gym_jumping_task/envs
    wget https://raw.githubusercontent.com/google-research/jumping-task/master/gym_jumping_task/__init__.py -O gym_jumping_task/__init__.py
    wget https://raw.githubusercontent.com/google-research/jumping-task/master/gym_jumping_task/envs/__init__.py -O gym_jumping_task/envs/__init__.py
    wget https://raw.githubusercontent.com/google-research/jumping-task/master/gym_jumping_task/envs/jumping_colors_task.py -O gym_jumping_task/envs/jumping_colors_task.py 
    wget https://raw.githubusercontent.com/google-research/jumping-task/master/gym_jumping_task/envs/jumping_coordinates_task.py -O gym_jumping_task/envs/jumping_coordinates_task.py
    wget https://raw.githubusercontent.com/google-research/jumping-task/master/gym_jumping_task/envs/jumping_task.py -O gym_jumping_task/envs/jumping_task.py
```

# these are needed for pyvirtualdisplay (to create a video of the episode)
``` bash
apt install -y xvfb x11-utils ffmpeg
```
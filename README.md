## How to run the programme?
### Packages
This programme runs with Python 3.8 and Tensorflow 2.8.0. [The OpenAI Gym environment for Super Mario Bros](https://github.com/Kautenja/gym-super-mario-bros) that we used in this programme is made by Christian Kauten.

For installation of Tensorflow 2.8.0, please refer to:
https://www.tensorflow.org/install/pip#ubuntu

For training with Nvdia GPUs, please refer to:
https://www.tensorflow.org/install/gpu

The following dependencies and PIP packages are required for this programme:
```
apt-get install ffmpeg libsm6 libxext6  -y
pip3 install gym
pip3 install gym-super-mario-bros
pip3 install opencv-python
```

### Run the programme
Within this project directory, please run
```
python3 main.py
```
for training the agent.


### plot reward graph
Within this project directory, please run
```
python3 plot_reward_epoch.py
```

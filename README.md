## 基本流程



- Python 3.7.16
- PyTorch 1.10.0+cu113
- Isaac Gym: Preview 4

1. Create an environment and install PyTorch:

  - `conda create -n himloco python=3.7.16`
  - `conda activate himloco`
  - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

2. Install Isaac Gym:
  - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym（下载安装包后解压，只运行下面的命令，不需要安装其他依赖）
  - `cd isaacgym/python && pip install -e .`

3. Clone this repository.

  - `git clone https://github.com/OpenRobotLab/HIMLoco.git`
  - `cd HIMLoco`


4. Install HIMLoco.
  - `cd rsl_rl && pip install -e .`
  - `cd ../legged_gym && pip install -e .`

**Note:** Please use legged_gym and rsl_rl provided in this repo, we have modefications on these repos.

### Tutorial

1. Train a policy:

  - `cd legged_gym/legged_gym/scripts`
  - `python train.py`
  
#### 会经历一些报错：
1.ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory
  - 解决方法：find /home /usr /opt -name "libpython3.7m.so.1.0" 2>/dev/null，找到路径，没有就下载；
  - 找到路径后：
  - nano ~/.bashrc；
  - 把搜索到的路径添加到最后一行：export LD_LIBRARY_PATH=路径:$LD_LIBRARY_PATH。
  - source ~/.bashrc

2.没有安装tensorboard。
  - 解决：conda activate himloco ; pip install tensorboard

3.AttributeError: module 'distutils' has no attribute 'version'
  - 解决方法：在anaconda里找到himloco环境的torch源文件，一般是：/anaconda3/envs/himloco/lib/python3.7/site-packages/torch
  - 在torch文件夹的子文件夹utils中找到tensorboard文件夹，打开__init__.py
  - 找到代码：
  - from distutils import version（或：from distutils.version import LooseVersion）； 
  - LooseVersion = version.LooseVersion
  - 修改为：from packaging.version import Version；LooseVersion = Version;
  - 这样一般就可以解决因为版本导致的报错。


2. Play and export the latest policy:
  - `cd legged_gym/legged_gym/scripts`
  - `python play.py`


3. 参数修改：
  - HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py里是基本的配置；

  - 默认使用机器人：aliengo

  - 修改配置参数时不能只修改基类的配置，还要修改所使用机器人的相应配置。或者运行时直接使用命令行参数如： --num_envs=2048

  - wandb和tensorboard的记录都在HIMLoco/legged_gym/logs里。

  - play.py默认使用最后一次记录的训练权重，但是我试验发现4000~5000轮次之后性能可能发生急剧下降，因此查看结果时需要提前看一下什么时候收敛，在HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py基础配置的最后一个类中修改调用的权重。

  - rsl_rl强化学习库中带him_ 前缀的文件是HIMLoco (Hybrid Internal Model) 的实现版本,而不带前缀的是标准 PPO 的基础实现。

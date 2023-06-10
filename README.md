# IsaacGym-bot
my training programme for Ubot and EXOskeleton robots
这个项目不包含isaacGym的基本架构，需要通过下载https://github.com/NVIDIA-Omniverse/IsaacGymEnvs 来获得本程序的运行环境。
在将对应文件夹下的内容同步到本地仓库之后,找到train.py所在文件夹,使用如下命令开始训练：
python train.py task=Ubot
在训练完成后使用如下命令加载训练完成的策略
python train.py task=Ubot test=True checkpoint=...
其中checkpoint载入文件位于runs文件夹中
如果需要修改物理仿真参数，可以检查cfg文件夹下的task
如果需要修改强化学习参数，可以检查cfg文件夹下的train

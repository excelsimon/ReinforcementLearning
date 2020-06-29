# BipedalWalker-v3 
## Introduction
- Framework: **[PARL](https://github.com/PaddlePaddle/PARL)** 
- Algorithm: 
    - DDPG
    - SAC
- **[BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/)** is not supported by gym. 
  I implemented BipedalWalker-v3.
- box2d-py is needed to install.
   ``` pip install box2d-py```

# Result

## DDPG_v1
- The BipedalWalker can move forward to the far end with single leg, aha, BipedalWalker-V3.mp4 show this and it takes 27 seconds.
- After about 100w steps, the mean reward of 5 test episode reaches 51.498. 
<div align="left"><img src="./BipedalWalkerV3.gif"/></div>

## DDPG_V2
- The BipedalWalker can RUN quickly to the far end, BipedalWalker-V3_version2.mp4 show this and it shorten the time to 12 seconds.
- After about 70w episode, the mean reward of 5 test episode reaches 323.10
- The improvement is potentially because of one more fc layer is used and the size of each layer is twice as big as DDPG_v1.
Besides, once the bipedal walker failed, the reward is -100. Compared to -0.x reward, the -100 reward may do harm to the training process.
I change -100 to -5 when training and it proves to be beneficial.
<div align="left"><img src="./BipedalWalkerV3_version2.gif"/></div>

## SAC
- Also I tried soft actor-critic architecture and the reward reaches 0.5 within 20w steps.
- The training is going on.


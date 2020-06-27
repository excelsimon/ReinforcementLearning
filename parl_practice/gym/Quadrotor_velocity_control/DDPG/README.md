# Quadrotor velocity_control task 
## Introduction
- Framework: **[PARL](https://github.com/PaddlePaddle/PARL)** 
- Algorithm: DDPG
- **[Quadrotor](https://github.com/PaddlePaddle/RLSchool/tree/master/rlschool/quadrotor)** 
- **[rlschool](https://github.com/PaddlePaddle/RLSchool/tree/master/rlschool/quadrotor)** is needed.
   ``` pip install rlschool```

# Result
- 模型在10万步reward收敛到-100以内
- 模型在20万步收敛到-30以内
- 测试单episode最好结果-4.95
- 测试5个episode平均最好reward为-17.02
- 测试5个episode平均reward在-20左右波动
- Visit **[Quadroto velocity control](https://aistudio.baidu.com/aistudio/projectdetail/595032)** to get more info.


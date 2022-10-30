#### CPU vs GPU
I always get confused when my tensors should be on CPU and when on GPU.. The experience replay buffer is in RAM, right? Should I immediately move everything to GPU right after retrieving from the buffer? when i calculate the targets (targets = rewards + gamma * max(target_q_value, 1)) * (1-dones) should i move the rewards and dones to GPU or move the target_q_value from the network to CPU? 

Note: The targets are later used to calculate the loss. Can it be that the gradient graph is damaged when switching from GPU to CPU and back?


#### What is Q_t(f) in the paper
I understand how the action selection process can be thought of as a multi-armed bandit problem. 
However, in a multi-armed bandit problem i have the Reward from the environment that i try to maximize, what do i maximize in UCBDrAC?

> Q is computed as a sliding window average of the past K mean returns obtained by the agent after being updated using augmentation f
# Adversarial-MATL on a simple adversarial environment

Here, I am initially training 2  good agents and an adversary agent in the simple adversary environment of the pettingzoo, by using duelling double DQN algorithm.<br>
Source task: 2 good agents and an adversary<br>
Target task: 3 good agents and an adversary<br>
Pretraining is done on the source task.<br>
<br>
To do pretraining, run the code: python main.py --pretrain True<br>
To do transfer learning on the target task, run the code: python main.py --transfer_train True<br>
To learn the target task from scratch, run the code: python main.py<br>
<br>
On the target task, the following experiments are conducted. To compare the perfomance, all the agents are also learned from scratch in the very same environment:
1. Transfer learning is implimented on 2 of the 3 good agents and the adversary (corresonding agents from the source task) while the remaining agent is learnt from scratch. 
2. Transfer learning is done on all the good agents in the target task (by learning from the agent in the source task which had the best perfomance) and the adversary.

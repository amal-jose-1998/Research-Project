Adversarial-MATL on a Simple Adversarial Environment
In this project, I am initially training 2 good agents and an adversary agent in the simple adversary environment of the PettingZoo library, using the dueling double DQN algorithm.

Source Task:

2 good agents
1 adversary agent
Target Task:

3 good agents
1 adversary agent
Pretraining is performed on the source task.

Instructions:
To perform pretraining, run the following code:

css
Copy code
python main.py --pretrain True
To conduct transfer learning on the target task, run the code:

css
Copy code
python main.py --transfer_train True
To learn the target task from scratch, run the code:

css
Copy code
python main.py
Target Task Experiments:
On the target task, the following experiments are conducted. To compare performance, all agents are also learned from scratch in the same environment:

Experiment 1:

Transfer learning is implemented on 2 of the 3 good agents and the adversary (corresponding agents from the source task).
The remaining agent is learned from scratch.
Experiment 2:

Transfer learning is performed on all the good agents in the target task (learning from the agent in the source task with the best performance) and the adversary.
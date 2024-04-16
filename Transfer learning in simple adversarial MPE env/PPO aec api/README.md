# Adversarial-MATL on a simple adversarial environment with AEC API (PPO)

Gaining insights from the previous experiments (done on the ddDQN algorithm), here I am training the good agents and an adversary agent in the simple adversary environment of the pettingzoo, by using the PPO algorithm. All the agents are pretrained on the source task (2 agents and an adversary), and undergo coarse-to-fine tuning in the target task (3 agents and an adversary) <br><br>
Source task: 2 good agents and an adversary<br>
Target task: 3 good agents and an adversary<br>
<br>
Run the below commands for the respective implementations<br>
To do pretraining, run the code: `python main.py --pretrain True`<br>
To do transfer learning on the target task (for all the agents), run the code: `python main.py --transfer_train True --train_all_agents True --best_good_agent agent_1`, the best good agent should be chossen based on the loss curves obtained during pretraining<br>
To learn the target task from scratch, run the code: `python main.py`<br>
<br>


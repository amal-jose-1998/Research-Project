# Adversarial-MATL on a simple adversarial environment with AEC API (ddDQN)

Similar to the experiments done with parallel API, here too I am training 2  good agents and an adversary agent in the simple adversary environment of the pettingzoo, by using the duelling double DQN algorithm. Gaining insights from the previous experiments (done on the parallel API), transfer learning is done by learning all the agents from the source task. <br><br>
Source task: 2 good agents and an adversary<br>
Target task: 3 good agents and an adversary<br>
Pretraining is done on the source task.<br>
<br>
Run the below commands for the respective implementations (given as 2 seperate folders)<br>
To do pretraining, run the code: `python main.py --pretrain True`<br>
To do transfer learning on the target task (for all the agents), run the code: `python main.py --transfer_train True --train_all_agents True --best_good_agent 2`, the best good agent should be chossen based on the loss curves obtained during pretraining<br>
To learn the target task from scratch, run the code: `python main.py`<br>
<br>
On the target task, the following experiments are conducted.
1. Transfer learning is implimented by just employing the fine tuning. This means that the CNNs in the initial layers are left unfrozzen and learned with each learning iteration. This is provided in the folder named "only fine tuning".
2. Transfer learning is implemented by employing coarse tuning first and then fine tuning. That is, the CNNs are initially frozzen for any updation while the model takes comparatively larger learning steps. After some training iterations (50 in the code), the CNNs are unfrozzen and enabled for fine tuning, where the learning steps are smaller. This implementation is given in the folder named "coarse & fine tuning".

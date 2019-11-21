# Created: 21st of November 2019
# Author: Niels Pressel

import time
import os


class SessionManager:

    def __init__(self):
        self.id = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())
        self.folder_path = "../../data/" + self.id

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def write_session_info(self, lr, eps_decay, gamma, model, target_net_update, memory_size, batch_size, episodes,
                           max_steps_per_episode):
        with open(os.path.join(self.folder_path, "info.txt"), "w") as f:
            f.write("Learning rate: " + str(lr) + "\n")
            f.write("Gamma: " + str(gamma) + "\n")
            f.write("Epsilon decay: " + str(eps_decay) + "\n")
            f.write("Target network update: " + str(target_net_update) + "\n")
            f.write("Replay memory size: " + str(memory_size) + "\n")
            f.write("Learning batch size: " + str(batch_size) + "\n")
            f.write("Episode count: " + str(episodes) + "\n")
            f.write("Maximum steps per episode: " + str(max_steps_per_episode) + "\n\n")

            model.summary(print_fn=lambda x: f.write(x + "\n"))

    def write_score(self, scores):
        with open(os.path.join(self.folder_path, "scores.txt"), "w") as f:
            for item in scores:
                f.write(str(item) + ", ")

    def get_folder_path(self):
        return self.folder_path

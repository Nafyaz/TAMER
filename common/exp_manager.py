from tqdm import tqdm


class ExpManager:
    def __init__(self, train_env=None, eval_env=None, algo=None):
        self.train_env = train_env
        self.eval_env = eval_env
        self.algo = algo

    def train(self, step_count=1000, will_eval=True):
        if self.train_env is None or self.algo is None:
            raise Exception("Train environment or algorithm cannot be none")

        total_reward = 0
        eval_rewards = []

        observation, info = self.train_env.reset()

        for step in tqdm(range(step_count)):
            action = self.algo.predict(observation)
            new_observation, reward, terminated, truncated, info = self.train_env.step(action)
            total_reward += reward
            # print("reward: ", reward)
            # print("new state: ", new_observation)
            self.algo.update_model(observation, action, reward, new_observation)

            if will_eval:
                eval_reward = self.eval(episode_count=310)
                # print(f"eval after step {step}: {eval_reward}")
                eval_rewards.append(eval_reward)

            observation = new_observation

            if terminated or truncated:
                # print("Starting new episode for train")
                observation, info = self.train_env.reset()

        self.train_env.close()
        return total_reward, eval_rewards

    def eval(self, episode_count=None, step_count=None):
        if self.eval_env is None or self.algo is None:
            raise Exception("Eval environment or algorithm cannot be none")
        if (episode_count is None and step_count is None) or (episode_count is not None and step_count is not None):
            raise Exception("Either episode_count or step_count should be None")

        total_reward = 0
        observation, info = self.eval_env.reset()

        if step_count is not None:
            for _ in range(step_count):
                action = self.algo.predict(observation)
                # print("old state: ", observation)
                new_observation, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += reward
                # print("new state: ", new_observation)
                # print("reward: ", reward)
                # print("terminated: ", terminated)
                # print("truncated: ", truncated, end="\n\n")

                observation = new_observation

                if terminated or truncated:
                    observation, info = self.eval_env.reset()

            self.eval_env.close()
            return total_reward / step_count

        if episode_count is not None:
            for episode in range(episode_count):
                while True:
                    action = self.algo.predict(observation)
                    # print("episode: ", episode)
                    # print("old state: ", observation)
                    new_observation, reward, terminated, truncated, info = self.eval_env.step(action)
                    total_reward += reward
                    # print("new state: ", new_observation)
                    # print("reward: ", reward)
                    # print("terminated: ", terminated)
                    # print("truncated: ", truncated, end="\n\n")

                    observation = new_observation

                    if terminated or truncated:
                        observation, info = self.eval_env.reset()
                        break

            self.eval_env.close()
            return total_reward / episode_count

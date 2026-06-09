import gymnasium as gym
import numpy as np
import time

env = gym.make("CartPole-v1", render_mode="human")
(state,_) = env.reset()

num_of_episodes = 5
timeSteps=100

for episodeIndex in range(num_of_episodes):
    print(episodeIndex)
    initial_state=env.reset()
    env.render()
    appendedObservations=[]
    for timeIndex in range(timeSteps):
        print(timeIndex)
        random_action=env.action_space.sample()
        (observation, reward, terminated, truncated, info) = env.step(random_action)

        # next_state:
        # The new situ czcation of the environment after your action. For CartPole, it's the array of 4 numbers we talked about earlier (new cart position, new pole angle, etc.). You use this to decide your next action.
        # reward: A number representing how good that step was. For CartPole, you always get exactly 1.0 if the pole didn't fall over.
        # terminated: A boolean (True or False). It is True if you "lost" or "won" the game naturally. In CartPole, it becomes True the moment the pole tilts too far or the cart drives off the screen.
        # truncated: A boolean (True or False). It is True if the game was cut short by a time limit. For CartPole, this becomes True if you successfully survive for exactly 500 steps.
        # info: A dictionary containing extra diagnostic information or debugging details. Usually, you don't need to worry about this and can just ignore it.
        
        # just keep in mind what env.step() returns back
        
        appendedObservations.append(observation)
        time.sleep(0.1)
        if (terminated):
            time.sleep(1)
            break
env.close()   
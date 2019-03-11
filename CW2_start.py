from __future__ import division
import gym
from gym.envs.registration import register
import numpy as np
import pandas as pd
import random, math, time
import copy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
register(
    id          ='FrozenLakeNotSlippery-v0',
    entry_point ='gym.envs.toy_text:FrozenLakeEnv',
    kwargs      ={'map_name' : '8x8', 'is_slippery': False},
)

def running_mean(x, N=20):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

class Agent:
    def __init__(self, env):
        self.stateCnt      = 6 #一共有多少种状态
        self.actionCnt     = ['left', 'right', 'up','down']# left:0; down:1; right:2; up:3
        self.learning_rate = 0.4
        self.gamma         = 0.8  #未来的衰减值
        self.epsilon       = 0.8  #greedy police 80%的时候选择最优的动作，20%的时候选择随机的动作
        self.maxepisode = 13
        self.Q             = self._initialiseModel()

    def _initialiseModel(self, n_states, actions):
        table = pd.DataFrame(
            np.zeros((n_states, len(actions))),
            columns = actions,
        )
        print(table)
        return table

    #def predict_value(self, s):

    def update_value_Qlearning(self, s,a,r,s_next, goalNotReached):
        self.Q[s,a] = self.Q[s,a] + self.learning_rate * (r + self.gamma * np.max(self.Q[s_next,:]) - self.Q[s,a] )
        s = s_next
        if success == true:

            break

    #def update_value_SARSA(self, s,a,r,s_next, a_next, goalNotReached):

    def choose_action(self, state, q_table):
        state_actions = q_table.iloc[state,:]
        if(np.random.uniform() > epsilon) or (state_actions.all()==0):
            action_name = np.random.choice(actionCnt)
        else:
            action_name = state_actions.argmax() #选较大值
        return action_name

    def updateEpsilon(self, episodeCounter):


class World:
    def __init__(self, env):
        self.env = env
        print('Environment has %d states and %d actions.' % (self.env.observation_space.n, self.env.action_space.n))
        self.stateCnt           = stateCnt
        self.actionCnt          = actionCnt
        self.maxStepsPerEpisode = maxStepPerEpisode
        self.q_Sinit_progress   = q_Sinit_progress # ex: np.array([[0,0,0,0]])

    def run_episode_qlearning(self):
        s               = self.env.reset() # "reset" environment to start state
        r_total         = 0
        episodeStepsCnt = 0
        success         = False
        for i in range(self.maxStepsPerEpisode):
            # self.env.step(a): "step" will execute action "a" at the current agent state and move the agent to the nect state.
            # step will return the next state, the reward, a boolean indicating if a terminal state is reached, and some diagnostic information useful for debugging.
            # self.env.render(): "render" will print the current enviroment state.
            # self.q_Sinit_progress = np.append( ): use q_Sinit_progress for monitoring the q value progress throughout training episodes for all available actions at the initial state.
        return r_total, episodeStepsCnt

    def run_episode_sarsa(self):
        s               = self.env.reset() # "reset" environment to start state
        r_total         = 0
        episodeStepsCnt = 0
        success         = False
        for i in range(self.maxStepsPerEpisode):
            # self.env.step(a): "step" will execute action "a" at the current agent state and move the agent to the nect state.
            # step will return the next state, the reward, a boolean indicating if a terminal state is reached, and some diagnostic information useful for debugging.
            # self.env.render(): "render" will print the current enviroment state.
            # self.q_Sinit_progress = np.append( ): use q_Sinit_progress for monitoring the q value progress throughout training episodes for all available actions at the initial state
        return r_total, episodeStepsCnt

    def run_evaluation_episode(self):
        agent.epsilon = 0
        return success


if __name__ == '__main__':  #主运行函数
    env                      = gym.make('FrozenLakeNotSlippery-v0')
    world                    = World(env)
    agent                    = Agent(env) # This will creat an agent
    r_total_progress         = []
    episodeStepsCnt_progress = []
    nbOfTrainingEpisodes     =
    for i in range(nbOfEpisodes):
        print('\n========================\n   Episode: {}\n========================'.format(i))
        # run_episode_qlearning or run_episode_sarsa
        # append to r_total_progress and episodeStepsCnt_progress
    # run_evaluation_episode

    ### --- Plots --- ###
    # 1) plot world.q_Sinit_progress
    fig1 = plt.figure(1)
    plt.ion()
    plt.plot(world.q_Sinit_progress[:,0], label='left',  color = 'r')
    plt.plot(world.q_Sinit_progress[:,1], label='down',  color = 'g')
    plt.plot(world.q_Sinit_progress[:,2], label='right', color = 'b')
    plt.plot(world.q_Sinit_progress[:,3], label='up',    color = 'y')
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(prop = fontP, loc=1)
    plt.pause(0.001)

    # 2) plot the evolution of the number of steps per successful episode throughout training. A successful episode is an episode where the agent reached the goal (i.e. not any terminal state)
    fig2 = plt.figure(2)
    plt1 = plt.subplot(1,2,1)
    plt1.set_title("Number of steps per successful episode")
    plt.ion()
    plt.plot(episodeStepsCnt_progress)
    plt.pause(0.0001)
    # 3) plot the evolution of the total collected rewards per episode throughout training. you can use the running_mean function to smooth the plot
    plt2 = plt.subplot(1,2,2)
    plt2.set_title("Rewards collected per episode")
    plt.ion()
    r_total_progress = running_mean(r_total_progress)
    plt.plot(r_total_progress)
    plt.pause(0.0001)
    ### --- ///// --- ###

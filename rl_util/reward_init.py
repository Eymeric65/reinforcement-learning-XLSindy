"""
This file provides reward functions and initial conditions function for multiple environments and experiment
"""

import numpy as np

import jax.numpy as jnp


initial_state = np.array([[np.pi, 0], [np.pi, 0]])

def reward_1(state,action):

    action,state = action[0],state[0]

    position = state[::2]
    velocity = state[1::2]

    terminated = 0
    if position[0] < np.pi/2 or position[1] < np.pi/2 or position[0] > 3*np.pi/2 or position[1] > 3*np.pi/2:
        
        terminated = 1

    # Reward function to minimize energy while going upward
    # Reward for minimizing energy (kinetic + potential)
    kinetic_energy = (velocity[0] ** 2) + (velocity[1] ** 2)
    action_penalty =  -((action[0] ** 2) +  (action[1] ** 2))*0.001
    #potential_energy = mass1 * 9.81 * (1 - np.cos(position[0])) + mass2 * 9.81 * (1 - np.cos(position[1]))
    energy_penalty = kinetic_energy #+ potential_energy
    
    # Reward for moving upward
    #upward_reward = - (np.cos(position[0]) + np.cos(position[1]) - 2)
    upward_reward = - ( (np.pi-position[0])**2 +(np.pi-position[1])**2) + (np.pi)**2/2 # Shaping reward for positivity stay in 20 degree cone
    
    # Total reward
    terminated_penalty =-100 * terminated

    total_reward = upward_reward + action_penalty + terminated_penalty # -action_penalty*0.1
    
    return total_reward, terminated, {}

def reward_swing_up(
        mass1:float =1 ,
        mass2:float =1,
        lenght1:float =1,
        lenght2:float =1,
        max_energy:float = 5,
    ):
    """
    Reward function for the swing up problem.
    This function reward the agent for getting enough energy to swing up the pendulum
    """

    def reward(state,action):

        action,state = action[0],state[0]

        position = state[::2]
        velocity = state[1::2]

        reward_info={}

        terminated = 0

        if position[0] < -2*np.pi or position[1] < -2*np.pi or position[0] > 2*np.pi or position[1] > 2*np.pi:
        
            terminated = 1


        max_potential_energy = lenght1 * mass1 * 9.81 + lenght2 * mass2 * 9.81 

        potential_energy = lenght1 * mass1 * 9.81 * (1 - np.cos(position[0])) + lenght2 *mass2 * 9.81 * (1 - np.cos(position[1]))
        kinetic_energy = mass1 *(lenght1 *velocity[0] ** 2)/2 + (lenght2 *velocity[1] ** 2)*mass2/2

        total_energy = potential_energy + kinetic_energy

        # Main issue is dissipation forces...
        total_energy_ratio = total_energy/(max_potential_energy) # equal to 1 when the pendulum is at the top with energy surplus

        energy_reward = -abs(total_energy_ratio - max_energy) + max_energy # equal to 1 when the pendulum is at the top and diminishes if the energy get greater (discard infinite speed reward)
        
        # test with absoulte value instead of square
        upward_reward = - ( abs(1-position[0]/np.pi) +abs(1-position[1]/np.pi))/2 + 1 # Shaping function to give reward to reach the upward position =1 when reached
        
    
        action_penalty =  -(action[0] ** 2) -  (action[1] ** 2)

        velocity_penalty = - (velocity[0] ** 2) - (velocity[1] ** 2)

        reward_info['goal_state'] = upward_reward # This info is only used to know if agent can succeed the task It should be bounded between 0 and 1 and 1 would signify that the agent is at the top.

        # At this point all the reward are bounded between 0 and 1
        # we can apply goal priority to the reward

        if upward_reward > 0.7: # If the agent is going upward we can discard the energy reward and reward more the upward reward

            energy_reward = energy_reward*0.01
            #this work if the energy reward is already negligeable
            upward_reward = upward_reward*2


        # Apply scaling to the reward now, it helps for reading the info graph
        energy_reward = energy_reward*5
        upward_reward = upward_reward*30
        #action_penalty = action_penalty*0.005 # base value for double action swing up
        action_penalty = action_penalty*0.005
        velocity_penalty = velocity_penalty*0.03 # base value for double action swing up


        reward_info['energy_reward'] = energy_reward
        reward_info['upward_reward'] = upward_reward
        reward_info['action_penalty'] = action_penalty
        reward_info['velocity_penalty'] = velocity_penalty

        # The goal is to force the agent to maximize the energy while going upward
        # only action_penalty is unbounded and may induce infinite penalty that could slow down learning...
        total_reward = energy_reward + upward_reward + action_penalty + velocity_penalty - 1000*terminated 

        return total_reward, terminated ,reward_info
    
    return reward

def reward_swing_up_s(
        mass1:float =1 ,
        mass2:float =1,
        lenght1:float =1,
        lenght2:float =1,
        max_energy:float = 5,
    ):
    """
    Reward function for the swing up problem single acted.
    This function reward the agent for getting enough energy to swing up the pendulum
    """

    def reward(state,action):

        action,state = action[0],state[0]

        position = state[::2]
        velocity = state[1::2]

        reward_info={}

        terminated = 0

        if position[0] < -2*np.pi or position[1] < -2*np.pi or position[0] > 2*np.pi or position[1] > 2*np.pi:
        
            terminated = 1


        max_potential_energy = lenght1 * mass1 * 9.81 + lenght2 * mass2 * 9.81 

        potential_energy = lenght1 * mass1 * 9.81 * (1 - np.cos(position[0])) + lenght2 *mass2 * 9.81 * (1 - np.cos(position[1]))
        kinetic_energy = mass1 *(lenght1 *velocity[0] ** 2)/2 + (lenght2 *velocity[1] ** 2)*mass2/2

        total_energy = potential_energy + kinetic_energy

        # Main issue is dissipation forces...
        total_energy_ratio = total_energy/(max_potential_energy) # equal to 1 when the pendulum is at the top with energy surplus

        energy_reward = -abs(total_energy_ratio - max_energy) + max_energy # equal to 1 when the pendulum is at the top and diminishes if the energy get greater (discard infinite speed reward)
        
        # test with absoulte value instead of square
        upward_reward_1 = -  abs(1-position[0]/np.pi)  + 1 # Shaping function to give reward to reach the upward position =1 when reached
        upward_reward_2 = -  abs(1-position[1]/np.pi)  + 1 # Shaping function to give reward to reach the upward position =1 when reached
    
        upward_reward = (upward_reward_1*1 + upward_reward_2*4)/5 # reward more the non actuated pendulum

        action_penalty =  -(action[0] ** 2) -  (action[1] ** 2)

        velocity_penalty = - (velocity[0] ** 2) - (velocity[1] ** 2)

        reward_info['goal_state'] = upward_reward # This info is only used to know if agent can succeed the task It should be bounded between 0 and 1 and 1 would signify that the agent is at the top.

        # At this point all the reward are bounded between 0 and 1
        # we can apply goal priority to the reward

        if upward_reward > 0.7: # If the agent is going upward we can discard the energy reward and reward more the upward reward

            energy_reward = energy_reward*0.01
            #this work if the energy reward is already negligeable
            upward_reward = upward_reward*2


        # Apply scaling to the reward now, it helps for reading the info graph
        energy_reward = energy_reward*5
        upward_reward = upward_reward*30
        #action_penalty = action_penalty*0.005 # base value for double action swing up
        action_penalty = action_penalty*0.005
        velocity_penalty = velocity_penalty*0.03 # base value for double action swing up


        reward_info['energy_reward'] = energy_reward
        reward_info['upward_reward'] = upward_reward
        reward_info['action_penalty'] = action_penalty
        reward_info['velocity_penalty'] = velocity_penalty

        # The goal is to force the agent to maximize the energy while going upward
        # only action_penalty is unbounded and may induce infinite penalty that could slow down learning...
        total_reward = energy_reward + upward_reward + action_penalty + velocity_penalty - 1000*terminated 

        return total_reward, terminated ,reward_info
    
    return reward

def initial_function_f(initial_state): # Used for training the first working agent

    def init():
        return  np.reshape(initial_state, (1,-1))
    
    return  init

def initial_function_f_jax(initial_state): # Used for training the first working agent

    def init():
        return  jnp.reshape(initial_state, (1,-1))
    
    return  init

def initial_function_1(): # Used for training the first working agent

    disturbance = (np.random.rand(2)-0.5)*0.2
    #disturbance = np.array([0,0])
    disturbed = initial_state.copy()

    disturbed[0,0] += disturbance[0]
    disturbed[1,0] += disturbance[1]

    return  np.reshape(disturbed, (1,-1))

def initial_function_v(variance:float =0.2): # Used for training the first working agent

    def init():

        disturbance = (np.random.rand(2)-0.5)*variance
        #disturbance = np.array([0,0])
        disturbed = initial_state.copy()

        disturbed[0,0] += disturbance[0]
        disturbed[1,0] += disturbance[1]

        return  np.reshape(disturbed, (1,-1))
    
    return init
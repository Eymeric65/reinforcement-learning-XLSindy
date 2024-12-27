import numpy as np

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
    #action_penalty =  (mass1 * (action[0] ** 2) +  (action[1] ** 2))
    #potential_energy = mass1 * 9.81 * (1 - np.cos(position[0])) + mass2 * 9.81 * (1 - np.cos(position[1]))
    energy_penalty = kinetic_energy #+ potential_energy
    
    # Reward for moving upward
    #upward_reward = - (np.cos(position[0]) + np.cos(position[1]) - 2)
    upward_reward = - ( (np.pi-position[0])**2 +(np.pi-position[1])**2)
    
    # Total reward
    total_reward = upward_reward # -action_penalty*0.1
    
    return total_reward, terminated

def initial_function_1():

    disturbance = (np.random.rand(2)-0.5)*0.2
    #disturbance = np.array([0,0])
    disturbed = initial_state.copy()

    disturbed[0,0] += disturbance[0]
    disturbed[1,0] += disturbance[1]

    return  np.reshape(disturbed, (1,-1))
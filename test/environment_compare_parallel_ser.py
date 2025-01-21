"""
This script has been made to check if there is any anomaly between the serialized and parallel environment.

So far no anomaly has been detected for the runtime but the reward is totally broken 

"""



from rl_util import environment
import xlsindy
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from rl_util import agent
import torch
from rl_util import reward_init
import os

import jax.numpy as jnp
import jax

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initial parameters
link1_length = 1.0
link2_length = 1.0
mass1 = 0.8
mass2 = 0.8
initial_conditions = np.array([[0, 0], [0, 0]])  # Initial state matrix (k,2)
#friction_forces = [-1.4, -1.2]
friction_forces = [-0, -0]
# max_force_span = [15.8, 4.5]
# time_period = 1.0
# time_shift = 0.2
# scale_factor = 10  # Base multiplier for scaling
# num_periods = 5  # Number of periods for the simulation

# Symbols and symbolic matrix generation
time_sym = sp.symbols("t")
num_coordinates = 2
symbols_matrix = xlsindy.catalog_gen.generate_symbolic_matrix(num_coordinates, time_sym)

# Assign ideal model variables
theta1 = symbols_matrix[1, 0]
theta1_d = symbols_matrix[2, 0]
theta1_dd = symbols_matrix[3, 0]
theta2 = symbols_matrix[1, 1]
theta2_d = symbols_matrix[2, 1]
theta2_dd = symbols_matrix[3, 1]

m1, l1, m2, l2, g = sp.symbols("m1 l1 m2 l2 g")
# total_length = link1_length + link2_length
substitutions = {"g": 9.81, "l1": link1_length, "m1": mass1, "l2": link2_length, "m2": mass2}

# Lagrangian (L)
L = (0.5 * (m1 + m2) * l1 ** 2 * theta1_d ** 2 + 0.5 * m2 * l2 ** 2 * theta2_d ** 2 + m2 * l1 * l2 * theta1_d
     * theta2_d * sp.cos(theta1 - theta2) + (m1 + m2) * g * l1 * sp.cos(theta1) + m2 * g * l2 * sp.cos(theta2))

# Loop frequency
frequency = 25

dt = 1 / frequency

end_time = 9



model_path = os.path.abspath(
    "runs_parallel/rK4-DoublePendulum-v0__par_swing_up_double_action_2__1__1737433911/par_swing_up_double_action_2.cleanrl_model"
    )

# RL environment data generation

initial_state = np.array([0, 1, 0, 1])  # Initial state matrix (q0 ,q_d0 ,q1 ,q_d1)

parallel_env = 1

double_pendulum_environment_par = environment.Rk4Environment_parallel(
                                symbols_matrix,
                                time_sym,
                                L,
                                substitutions,
                                dt,
                                reward_function= reward_init.reward_swing_up_s_jax(),
                                fluid_forces=friction_forces,
                                initial_function=reward_init.initial_function_f_jax(initial_state),
                                max_time=end_time,
                                mask_action=np.array([1.0,1.0]),
                                action_multiplier=5.0,
                                parallel_envs=parallel_env)

double_pendulum_environment_ser = environment.Rk4Environment(
                                symbols_matrix,
                                time_sym,
                                L,
                                substitutions,
                                dt,
                                reward_function= reward_init.reward_swing_up_s(),
                                fluid_forces=friction_forces,
                                initial_function=reward_init.initial_function_f(initial_state),
                                max_time=end_time,
                                mask_action=np.array([[1.0,1.0]]),
                                action_multiplier=5.0)

agent = agent.Agent(double_pendulum_environment_par,model_path=model_path).to(device)

state_par = []
state_ser = []

action_arr = []

t_array = []

reward_arr_par = []
reward_arr_ser = []

t=0

double_pendulum_environment_par.init()
double_pendulum_environment_ser.reset()

#print("action shape : ",double_pendulum_environment_par.action_space.shape)

#print("shape action : ",jnp.zeros( double_pendulum_environment_par.action_space.shape+(2,)).shape )

# key = jax.random.PRNGKey(42)
# random_array = jax.random.uniform(key, shape= double_pendulum_environment_par.action_space.shape+(parallel_env,))

#print("random array :",random_array)

#print ("debug reset : ", double_pendulum_environment_par.step(random_array) ) # reset

#exit() # End of script 

start_time_p = time.perf_counter()

while t < end_time :

    #print(double_pendulum_environment_par.system_state,double_pendulum_environment_par.system_state.shape)
    
    with torch.no_grad():
        system_state_numpy = np.array(double_pendulum_environment_par.system_state)
        #print(system_state_numpy)
        action, _, _, _ = agent.get_action_and_value(torch.from_numpy(system_state_numpy).float().to(device))

    system_state_par, reward_par, terminated_par, truncated_par, info_par = double_pendulum_environment_par.step(action.cpu().numpy())

    #print(action.cpu().numpy())

    system_state_ser, reward_ser, terminated_ser, truncated_ser, info_ser = double_pendulum_environment_ser.step(action.cpu().numpy())
    #system_state_ser = system_state_ser+1
    #system_state, reward, terminated, truncated, info = double_pendulum_environment_par.step(np.ones((parallel_env,2))*0.5)
    #print("time   : ",double_pendulum_environment_par.t)
    #print("t      : ",t)
    #print("is terminated : ",terminated , truncated)

    #print(system_state.shape)
    #[:,::2] Store all information

    """ 
    What are the same so far :
    - position
    - velocity
    - goal_state
    - energy_reward

    FUCKKKKKKKKKKK i let an error in the upward_reward conversion 
    """

    print("debug_serial   :",info_ser["reward_info"][0]["upward_reward"])

    print("debug_parallel :",info_par["reward_info"]["upward_reward"])

    t+=dt

    state_par += [system_state_par]
    reward_arr_par += [reward_par]

    state_ser += [system_state_ser]
    reward_arr_ser += [reward_ser]

    #t_array += [double_pendulum_environment_par.t]
    t_array += [t]


end_time_p = time.perf_counter()

total_time = end_time_p-start_time_p


state_par = np.array(state_par)
state_ser = np.array(state_ser)

reward_arr_par = np.array(reward_arr_par)
reward_arr_ser = np.array(reward_arr_ser)

t_array = np.array(t_array)

print(f"""the computation of a simulation of :
        {end_time*parallel_env:.2f} s 
        {end_time*parallel_env/60:.2f} mn 
        {end_time*parallel_env/60/60:.2f} h 
        {end_time*parallel_env/60/60/24:.2f} days \n
        lasted {total_time:.2f} s on a computer using {parallel_env} parallel environments.
        This is equivalent to {total_time*1000000000/len(t_array)/parallel_env:.2f} ns / timestep""")



# reward_arr = np.array(reward_arr)

subject = 0

print(reward_arr_par.shape, reward_arr_ser.shape)

plt.figure()
plt.plot(t_array,reward_arr_par,label='reward_par')
plt.plot(t_array,reward_arr_ser,label='reward_ser')

plt.legend()

plt.figure()
plt.plot(t_array,state_par[:,subject,0],label='theta1_rl_par')
plt.plot(t_array,state_ser[:,0,0],label='theta1_rl_ser')

plt.legend()


plt.figure()
plt.plot(t_array,state_par[:,subject,0],label='theta1_rl_par')
plt.plot(t_array,state_ser[:,0,0],label='theta1_rl_ser')

plt.legend()

plt.figure()
plt.plot(t_array,state_par[:,subject,1],label='theta2_rl_par')
plt.plot(t_array,state_ser[:,0,1],label='theta2_rl_ser')

plt.legend()

plt.figure()
plt.plot(t_array,state_par[:,subject,2],label='theta2_rl_par')
plt.plot(t_array,state_ser[:,0,2],label='theta2_rl_ser')

plt.legend()

plt.figure()
plt.plot(t_array,state_par[:,subject,3],label='theta2_rl_par')
plt.plot(t_array,state_ser[:,0,3],label='theta2_rl_ser')

plt.legend()


# plt.figure()
# plt.plot(t_array,state[:,1,0],label='theta1_rl2')
# plt.plot(t_array,state[:,1,2],label='theta2_rl2')

# plt.legend()



# # plt.figure()
# # plt.plot(t_array, reward_arr[:, 0], label='reward')
# # plt.plot(t_array, reward_arr[:, 1], label='terminated')
# # plt.legend()

# action_arr = np.array(action_arr)


# plt.figure()
# plt.plot(t_array, action_arr[:,subject, 0], label='action1')
# plt.plot(t_array, action_arr[:,subject, 1], label='action2')

# plt.legend()

# xlsindy.render.animate_double_pendulum(link1_length,link2_length,state[:,subject,:],t_array)

plt.show() 




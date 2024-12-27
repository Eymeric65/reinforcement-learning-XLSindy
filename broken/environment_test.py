from rl_util import environment
import xlsindy
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

"""
This is just a test file to check the environment and the simulation of the double pendulum and compare it with the normal simulation.
"""

# Initial parameters
link1_length = 1.0
link2_length = 1.0
mass1 = 0.8
mass2 = 0.8
initial_conditions = np.array([[np.pi/2, 0], [np.pi/2, 0]])  # Initial state matrix (k,2)
friction_forces = [-1.4, -1.2]

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
frequency = 100

dt = 1 / frequency

end_time = 100

def initial_function():
    return  np.reshape(initial_conditions, (1,-1))

# RL environment data generation

double_pendulum_environment = environment.Rk4Environment(
                        symbols_matrix,
                        time_sym,
                        L,
                        substitutions,
                        dt,
                        reward_function= lambda x,y:(0,0),
                        initial_function=initial_function,
                        reset_overtime=False,)

double_pendulum_environment.reset()

state = []

t_array = []

while double_pendulum_environment.t < end_time :

    system_state, reward, terminated, truncated, info = double_pendulum_environment.step(np.array([0,0]))

    position = system_state[::2]

    state += [position]

    t_array += [double_pendulum_environment.t]

state = np.array(state)
t_array = np.array(t_array)

## Normal Xl sindy simulation generation

acceleration_func, _ = xlsindy.euler_lagrange.generate_acceleration_function(L, symbols_matrix, time_sym, substitution_dict=substitutions) #, fluid_forces=friction_forces)
dynamics_system = xlsindy.dynamics_modeling.dynamics_function(acceleration_func, lambda x:[0 , 0])

time_values, phase_values = xlsindy.dynamics_modeling.run_rk45_integration(dynamics_system, initial_conditions, end_time, max_step=0.01)
theta_values = phase_values[:, ::2]

##-------------------------------------

plt.plot(time_values,theta_values[:,0],label='theta1')
plt.plot(time_values,theta_values[:,1],label='theta2')

plt.plot(t_array,state[:,0],label='theta1_rl')
plt.plot(t_array,state[:,1],label='theta2_rl')

plt.legend()

plt.show() 




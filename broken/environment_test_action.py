"""
This is just a test file to check the environment and the simulation of the double pendulum and compare it with the normal simulation.
"""



from rl_util import environment
import xlsindy
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Initial parameters
link1_length = 1.0
link2_length = 1.0
mass1 = 0.8
mass2 = 0.8
initial_conditions = np.array([[0, 0], [0, 0]])  # Initial state matrix (k,2)
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

# RL environment data generation

double_pendulum_environment = environment.Rk4Environment(
                                                    symbols_matrix,
                                                    time_sym,
                                                    L,
                                                    substitutions,
                                                    dt,
                                                    reward_function= lambda x:(0,0),
                                                    fluid_forces=friction_forces,
                                                    reset_overtime=False)

res = double_pendulum_environment.reset(initial_conditions)

state = []

t_array = []

while double_pendulum_environment.t < end_time :

    print(double_pendulum_environment.t)

    system_state, reward, terminated, truncated, info = double_pendulum_environment.step(np.array([2,0]))

    position = system_state[::2]

    state += [position]

    t_array += [double_pendulum_environment.t]

state = np.array(state)
t_array = np.array(t_array)

plt.plot(t_array,state[:,0],label='theta1_rl')
plt.plot(t_array,state[:,1],label='theta2_rl')

plt.legend()

plt.show() 




import xlsindy
import sympy
import numpy as np
from typing import Dict,Callable,Tuple

class Rk4Environment:

    def __init__(
            self,
            symbols_matrix: np.ndarray,
            time_symbol:sympy.Symbol,
            lagrangian_formula:sympy.Expr,
            substitution_dict: Dict[str, float],
            dt:float,
            reward_function:Callable[[np.ndarray],Tuple[float,bool]]):

        self.reward_function = reward_function

        self._acceleration_func,_ = xlsindy.euler_lagrange.generate_acceleration_function(lagrangian_formula,symbols_matrix,time_symbol,substitution_dict=substitution_dict)
        # This following function construction assume that the action taken will define the force for the whole timestep (step action function)
        self.dynamics_function = xlsindy.dynamics_modeling.dynamics_function_fixed_external(self._acceleration_func) 
        self.dt = dt

        self.t = None
        self.system_state = None


    def reset(self,initial_state:np.ndarray):

        self.t = 0
        self.system_state = np.reshape(initial_state, (-1,))

    def step(self,action:np.ndarray):

        if(self.t is None or self.system_state is None):
            raise NotImplementedError("Need to run environment reset()")

        self.system_state = self._rk4_step(self.dynamics_function(action),self.t,self.system_state,self.dt)
        self.t = self.t + self.dt
        
        reward ,terminated =self.reward_function(self.system_state)

        info = None # Not implemented
        truncated = None # Not implemented

        return self.system_state, reward, terminated, truncated, info

    def _rk4_step(self,func, t, y, dt):
        k1 = func(t, y)
        k2 = func(t + dt / 2, y + dt / 2 * k1)
        k3 = func(t + dt / 2, y + dt / 2 * k2)
        k4 = func(t + dt, y + dt * k3)
        return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


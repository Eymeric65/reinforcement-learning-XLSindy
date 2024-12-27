import xlsindy
import sympy
import numpy as np
from typing import Dict,Callable,Tuple,List
from .spaces import BoxArray

class Rk4Environment:

    def __init__(
            self,
            symbols_matrix: np.ndarray,
            time_symbol:sympy.Symbol,
            lagrangian_formula:sympy.Expr,
            substitution_dict: Dict[str, float],
            dt:float,
            reward_function:Callable[[np.ndarray],Tuple[float,bool]],
            fluid_forces: List[int] = [],
            max_time:float = 60,
            initial_function:Callable[[],np.ndarray] = None,
            reset_overtime:bool = True):
        
        if(initial_function is None):
            raise NotImplementedError("Initial function not implemented")

        self.observation_space = BoxArray(shape=(1,symbols_matrix.shape[1]*2))
        self.action_space = BoxArray(shape=(1,symbols_matrix.shape[1]))

        self.reward_function = reward_function

        self._acceleration_func,_ = xlsindy.euler_lagrange.generate_acceleration_function(lagrangian_formula,symbols_matrix,time_symbol,substitution_dict=substitution_dict,fluid_forces=fluid_forces)
        # This following function construction assume that the action taken will define the force for the whole timestep (step action function)
        self.dynamics_function = xlsindy.dynamics_modeling.dynamics_function_fixed_external(self._acceleration_func) 
        self.dt = dt

        self.reset_overtime = reset_overtime

        self.initial_function = initial_function

        self.t = None
        self.max_time = max_time
        self.system_state = None


    #def reset(self,initial_state:np.ndarray):
    def reset(self):

        self.t = 0
        #self.system_state = np.reshape(initial_state, (1,-1)) # enforcing shape 
        self.system_state = self.initial_function()

        return self.system_state

    def step(self,action:np.ndarray):

        if(self.t is None or self.system_state is None):
            raise NotImplementedError("Need to run environment reset()")

        info = {} # Not implemented

        if(self.t >= self.max_time and self.reset_overtime):
            self.reset()
            return self.system_state, 0, [0], [1], info

        SS = self._rk4_step(self.dynamics_function(action),self.t,self.system_state,self.dt)

        self.system_state = SS
        self.t = self.t + self.dt
        
        reward ,terminated =self.reward_function(self.system_state,action)

        if terminated:
            self.reset()
        

        return SS, reward, [terminated], [0], info

    def _rk4_step(self,func, t, y, dt):
        k1 = func(t, y)
        k2 = func(t + dt / 2, y + dt / 2 * k1)
        k3 = func(t + dt / 2, y + dt / 2 * k2)
        k4 = func(t + dt, y + dt * k3)
        return np.reshape(y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), (1,-1)) # enforcing shape 


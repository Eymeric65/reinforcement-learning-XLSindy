import xlsindy
import sympy
import numpy as np
from typing import Dict,Callable,Tuple,List
from .spaces import BoxArray

# Jax speed up imports
import jax.numpy as jnp
from jax import jit
from jax import vmap
from jax import lax
import jax

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
            reset_overtime:bool = True,
            mask_action = np.array([[1.0,1.0]]),
            action_multiplier = 1.0,
            ):
        
        if(initial_function is None):
            raise NotImplementedError("Initial function not implemented")

        self.mask_action = mask_action
        self.action_multiplier = action_multiplier

        self.observation_space = BoxArray(shape=(1,symbols_matrix.shape[1]*2))
        self.action_space = BoxArray(shape=(1,symbols_matrix.shape[1]))

        self.reward_function = reward_function

        self._acceleration_func,_ = xlsindy.euler_lagrange.generate_acceleration_function(lagrangian_formula,symbols_matrix,time_symbol,substitution_dict=substitution_dict,fluid_forces=fluid_forces)
        # This following function construction assume that the action taken will define the force for the whole timestep (step action function)
        self.dynamics_function = xlsindy.dynamics_modeling.dynamics_function_fixed_external(self._acceleration_func) 
        self.dt = dt

        self.reset_overtime = reset_overtime

        self.initial_function = initial_function


        self.max_time = max_time

        self.system_state = None
        self.t = None
        self.total_reward = None

        self.key = 42 # the random key 


    #def reset(self,initial_state:np.ndarray):
    def reset(self):

        self.t = 0
        #self.system_state = np.reshape(initial_state, (1,-1)) # enforcing shape 
        self.system_state = self.initial_function()
        self.total_reward = 0

        return self.system_state

    def step(self,action:np.ndarray):

        action = np.array(action) * self.mask_action # scaling action

        if(self.t is None or self.system_state is None):
            raise NotImplementedError("Need to run environment reset()")

        info = {} #init info

        if(self.t >= self.max_time and self.reset_overtime):
            
            
            final_info = {
                'episode':{
                    'r':self.total_reward,
                    'l':self.t
                }
            }

            info["final_info"] = [final_info]
            self.reset()
            return self.system_state, 0, [0], [1], info

        SS = self._rk4_step(self.dynamics_function(action*self.action_multiplier),self.t,self.system_state,self.dt)

        self.system_state = SS
        self.t = self.t + self.dt
        
        reward ,terminated,reward_info =self.reward_function(self.system_state,action)

        info["reward_info"] = [reward_info]

        self.total_reward += reward

        if terminated:
            
            final_info = {
                'episode':{
                    'r':self.total_reward,
                    'l':self.t
                }
            }

            info["final_info"] = [final_info]
            self.reset()
        

        return SS, reward, [terminated], [0], info

    def _rk4_step(self,func, t, y, dt):
        k1 = func(t, y)
        k2 = func(t + dt / 2, y + dt / 2 * k1)
        k3 = func(t + dt / 2, y + dt / 2 * k2)
        k4 = func(t + dt, y + dt * k3)
        return np.reshape(y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), (1,-1)) # enforcing shape 

    # def _rk4_step_f(self,func, forces, y, dt):
    #     k1 = func(y, forces)
    #     k2 = func(y + dt / 2 * k1, forces)
    #     k3 = func( y + dt / 2 * k2, forces)
    #     k4 = func(y + dt * k3, forces)
    #     return np.reshape(y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), (1,-1)) # enforcing shape 
    

def rk4_step_f(func, forces, y, dt):
        
        k1 = func(y, forces)
        k2 = func(y + dt / 2 * k1, forces)
        k3 = func( y + dt / 2 * k2, forces)
        k4 = func(y + dt * k3, forces)
        return jnp.reshape(y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), (-1,)) # enforcing shape 

def step_creator(
        max_time:float,
        initial_function:Callable[[],jnp.ndarray],
        reward_function:Callable[[jnp.ndarray],Tuple[float,bool]],
        dynamics_function:Callable[[jnp.ndarray,jnp.ndarray],jnp.ndarray],
        integrator_function:Callable[[Callable[[jnp.ndarray,jnp.ndarray],jnp.ndarray],jnp.ndarray,jnp.ndarray,float],jnp.ndarray],
        reset_overtime:bool = True,
        dt:float = 0.01
):
    """
    Use this function to create a step function for the environment.
    The output function will be jited by jax. doesn't take in account action transformation. (action_multiplier and mask_action). 
    In order to streamline the jiting should 
    """
    
    def _step(
            system_state:jnp.ndarray, # need to be returned
            action:jnp.ndarray, 
            t:float, # need to be returned
            total_reward:float, # need to be returned
            key:int, # the fucking random key
    ):

        #print("debug inside _step : system_reward ",system_state,", action :",action,", t : ",t,", total_reward : ",total_reward)
    
        def reset(data):
            system_state, reward, [terminated], [truncated], info, t, total_reward ,key = data

            t = 0.0
            system_state,key = initial_function(key)
            total_reward = 0.0
            
            return system_state, reward, [terminated], [truncated], info, t, total_reward ,key
        
        def truncated_branch(data):
            system_state, reward, [terminated], [truncated], info, t, total_reward, key= data

            truncated = 1

            final_info = {
                'episode':{
                    'is_final':True,
                    'r':total_reward,
                    'l':t
                }
            }

            info["final_info"] = final_info

            system_state, reward, [terminated], [truncated], info, t, total_reward, key = reset((system_state, reward, [terminated], [truncated], info, t, total_reward, key))
            
            return system_state, reward, [terminated], [truncated], info, t, total_reward , key
        #if(t >= max_time and reset_overtime):

        def terminated_branch(data):
            system_state, reward, [terminated], [truncated], info, t, total_reward, key = data
            final_info = {
                'episode':{
                    'is_final':True,
                    'r':total_reward,
                    'l':t
                }
            }

            info["final_info"] = final_info

            system_state, reward, [terminated], [truncated], info, t, total_reward, key = reset((system_state, reward, [terminated], [truncated], info, t, total_reward, key))


            return system_state, reward, [terminated], [truncated], info, t, total_reward, key
        #if terminated:

        def normal_branch(data):

            system_state, reward, [terminated], [truncated], info, t, total_reward, key = data

            final_info = { # super quick fix, should be managed more efficiently
            'episode':{
                'is_final':False,
                'r':total_reward,
                'l':t
            }
            }

            info["final_info"] = final_info

            return system_state, reward, [terminated], [truncated], info, t, total_reward, key

        info = {} #init info
        truncated = 0
        terminated = 0

        #print("detect anomaly _a : ",system_state)

        system_state = integrator_function(dynamics_function,action,system_state,dt)
        t= t + dt

        #print("detect anomaly _p : ",system_state)

        reward ,terminated,reward_info = reward_function(system_state,action)

        info["reward_info"] = reward_info

        total_reward += reward

        #print("debug boolean", t >= max_time)
        #print("debug boolean", reset_overtime)

        branch_condition = (t >= max_time) & (reset_overtime)

        branch_index =  jnp.where(branch_condition, 0,  # Truncated
                        jnp.where(terminated, 1,  # Terminated
                                    2))  # Normal

        data = system_state, reward, [terminated], [truncated], info, t, total_reward, key

        return lax.switch(branch_index,[truncated_branch, terminated_branch, normal_branch],data)
    
    return _step

class SlidingMeanStandard:
    """
    Class in order to create a sliding average and standard average of some value.
    Practically this class will concern reward and observation
    """

    def __init__(
            self,
            batch_axis:int = None,
            moving_proportion:float = None,
            shape:tuple= None, # unused determined on the first update
            epsilon:float = 10e-8
    ):
        self.shape = shape
        self.moving_proportion = moving_proportion
        self.batch_axis = batch_axis
        self.epsilon = epsilon

        self.global_n=0

        self.mean = None
        self.var = None

    def update(self,batch):
        """
        Update the sliding mean and var over a new batch
        """
        if (self.batch_axis is not None):
            batch_size = batch.shape[self.batch_axis]
        else : 
            batch_size=np.prod(batch.shape)

        self.global_n += batch_size

        if(self.global_n == batch_size ):
            self.mean = np.mean(batch,axis = self.batch_axis)
            self.var = np.var(batch,axis = self.batch_axis)
        else :
            batch_mean = np.mean(batch,axis = self.batch_axis)
            batch_var = np.var(batch,axis = self.batch_axis)

            proportion = batch_size/self.global_n

            if(self.moving_proportion is not None and (proportion < self.moving_proportion)):
                proportion = self.moving_proportion

            diff_mean = batch_mean - self.mean

            self.mean = self.mean + diff_mean*proportion 

            self.var = self.var*(1-proportion) + batch_var*proportion + diff_mean**2 *proportion*(1-proportion)
        
        # Increment the global step

        return (batch - self.mean)/(self.var**0.5 + self.epsilon)

    def normalise(self,batch):

        return (batch - self.mean)/(self.var**0.5 + self.epsilon)


class Rk4EnvironmentParallel:
    """ 
    A modified version of the Rk4Environment class that allows for parallel environments.
    """

    """
    Recap of the error that i solved : 
    -Condition should be exprimed in a less pythonic way (with where, cond, etcc) ie : should be explained in a matrix possible way
    -Branched stuff should have the exactly same output (even when it output a dictionnary, the dictionnary should be exactly the same )
    """

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
            reset_overtime:bool = True,
            mask_action = np.array([[1.0,1.0]]),
            action_multiplier = 1.0,
            parallel_envs = 1,
            normalised_obs_reward = False,
            moving_proportion= 0.01
            ):
        """
        Initialize the environment with a plethora of parameters. Much attention should be directed to the initial and reward function that should be jax compatible.
        
        """
        
        if(initial_function is None):
            raise NotImplementedError("Initial function not implemented")

        self.parallel_envs = parallel_envs

        self.mask_action = jnp.reshape(mask_action,(-1))
        self.action_multiplier = action_multiplier

        self.observation_space = BoxArray(shape=(symbols_matrix.shape[1]*2,))
        self.action_space = BoxArray(shape=(symbols_matrix.shape[1],))

        self.reward_function = reward_function

        self._acceleration_func,_ = xlsindy.euler_lagrange.generate_acceleration_function(
                                                                                    lagrangian_formula,
                                                                                    symbols_matrix,
                                                                                    time_symbol,
                                                                                    substitution_dict=substitution_dict,
                                                                                    fluid_forces=fluid_forces,
                                                                                    lambdify_module="jax")
        # This following function construction assume that the action taken will define the force for the whole timestep (step action function)
        self.dynamics_function = xlsindy.dynamics_modeling.dynamics_function_RK4_env(
                                                                                    self._acceleration_func) 
        self.dt = dt

        self.reset_overtime = reset_overtime

        self.initial_function = initial_function


        self.max_time = max_time

        self.system_state = None
        self.t = None
        self.total_reward = None

        self.key =None # the random fucking key

        self.normalised_obs_reward = normalised_obs_reward

        # Normalisation
        self.RewardSlidingMeanStd = SlidingMeanStandard(
            batch_axis=None, # test change here 20250124
            moving_proportion=moving_proportion,
        )

        self.ObservationSlidingMeanStd = SlidingMeanStandard(
            batch_axis=None, # test change here 20250124
            moving_proportion=moving_proportion,
        )


        step_f = step_creator(
            max_time,
            initial_function,
            reward_function,
            self.dynamics_function,
            rk4_step_f,
            reset_overtime = reset_overtime,
            dt = self.dt
        )

        step_f = vmap(
                    step_f,
                    in_axes=(0,0,0,0,0),
                    out_axes=(0,0,0,0,0,0,0,0)
                )
        
        initial_function = vmap(
            initial_function,
            in_axes=(0,),
            out_axes=(0,0)
        )

        self.initial_function = initial_function
        
        self._step = jit(step_f)

        #resetting all environments

        
    def step(self,action:np.ndarray):

        action = np.array(action) * self.mask_action * self.action_multiplier# scaling action

        #print("shape of system : ",self.system_state.shape,action.shape,self.t.shape,self.total_reward.shape)

        self.system_state,reward,terminated,truncated,info,self.t,self.total_reward,self.key = self._step(self.system_state,action,self.t,self.total_reward,self.key)

        if self.normalised_obs_reward:
            self.system_state = self.ObservationSlidingMeanStd.update(self.system_state)   
            reward = self.RewardSlidingMeanStd.update(reward)*10       # Test scaling here 20250124

        return self.system_state, reward, terminated, truncated, info
    
    def init(self): # Need to be only once

        self.t = jnp.zeros(self.parallel_envs)
        self.total_reward = jnp.zeros(self.parallel_envs)

        key = jax.random.PRNGKey(42)

        print("first key : ",key)

        self.key = jax.random.split(key,num=self.parallel_envs) # can be changed to get other randomness

        print(self.key,self.key[0])

        #self.system_state = jnp.array([self.initial_function() for _ in range(self.parallel_envs)]) # .transpose()


        self.system_state,self.key = self.initial_function(self.key)

        if self.normalised_obs_reward:
            self.system_state = self.ObservationSlidingMeanStd.update(self.system_state)


        return self.system_state
        #print("init system state : ",self.system_state.shape)
        #print("t : ",self.t)








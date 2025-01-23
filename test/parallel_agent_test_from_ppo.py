from dataclasses import dataclass
from dataclasses import field

import os
import tyro

import time
import torch
from torch.utils.tensorboard import SummaryWriter 
import torch.optim as optim

import random 
import numpy as np

import xlsindy
import sympy as sp

from rl_util import environment
from rl_util import agent
from rl_util import reward_init

from typing import List

"""
Greatly inspired from clean RL continuous PPO
"""

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    #capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "rK4-DoublePendulum-v0"
    """the id of the environment"""
    total_timesteps: int = 2000000
    """total timesteps of the experiments"""
    learning_rate: float = 2e-3
    """the learning rate of the optimizer"""
    #num_envs: int = 1
    #"""the number of parallel game environments"""
    num_steps: int = 4096
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 254
    """the number of mini-batches"""
    update_epochs: int = 15
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    parallel_envs:int = 1 
    """the number of parallel rk4 environment"""

    #RK4 specific arguments
    frequency: int = 25
    """the frequency of the loop"""
    frame_skip:int = 1
    """number of framed skippped"""
    reward_function: str ="reward_swing_up_s()"
    """the reward function to be used"""
    init_function:str = "initial_function_f(np.array([[0, 0], [0, 0]]))"
    """the initial function to be used"""
    mask_action: List[float] =  field(default_factory=lambda: [1.0,0.0])
    """the mask action to be used"""
    friction_forces: List[float] =  field(default_factory=lambda: [-0.0, -0.0])
    """the friction forces to be used"""
    action_multiplier: float = 10.0
    """the action multiplier to be used"""
    model_path:str = None
    """the agent to continue to train"""
    
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.parallel_envs *args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    # Environment setup

    # Initial parameters
    link1_length = 0.5
    link2_length = 0.5
    mass1 = 1
    mass2 = 1
    #initial_state = np.array([[np.pi, 0], [np.pi, 0]])  # Initial state matrix (k,2)
    friction_forces = args.friction_forces
    #friction_forces = [-0.0, -0.0]

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
    substitutions = {"g": 9.81, "l1": link1_length, "m1": mass1, "l2": link2_length, "m2": mass2}

    # Lagrangian (L)
    L = (0.5 * (m1 + m2) * l1 ** 2 * theta1_d ** 2 + 0.5 * m2 * l2 ** 2 * theta2_d ** 2 + m2 * l1 * l2 * theta1_d
        * theta2_d * sp.cos(theta1 - theta2) + (m1 + m2) * g * l1 * sp.cos(theta1) + m2 * g * l2 * sp.cos(theta2))

    # Loop frequency
    frequency = args.frequency

    frame_skip= args.frame_skip

    dt = 1 / frequency
    # End of creation of double pendulum environment

    reward_function = eval(f"reward_init.{args.reward_function}")
    initial_function = eval(f"reward_init.{args.init_function}")
    mask_action = np.array([args.mask_action])

    #parallel_env = 10000
    parallel_env = args.parallel_envs

    env = environment.Rk4Environment_parallel(
                                    symbols_matrix,
                                    time_sym,
                                    L,
                                    substitutions,
                                    dt,
                                    reward_function= reward_function,
                                    fluid_forces=friction_forces,
                                    initial_function=initial_function,
                                    max_time=6,
                                    mask_action=mask_action,
                                    action_multiplier=args.action_multiplier,
                                    parallel_envs=parallel_env)

    agent = agent.Agent(env,model_path=args.model_path).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, parallel_env) + env.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, parallel_env) + env.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, parallel_env)).to(device)
    rewards = torch.zeros((args.num_steps, parallel_env)).to(device)
    dones = torch.zeros((args.num_steps, parallel_env)).to(device)
    values = torch.zeros((args.num_steps, parallel_env)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    #next_obs = env.reset(initial_state)
    next_obs = np.array(env.init())
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)


    for iteration in range(1,args.num_iterations +1):
        
        # Annealing the learning rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lr_now = args.learning_rate * frac
            optimizer.param_groups[0]["lr"] = lr_now

        t_arr = []
        t=0

        for step in range(0,args.num_steps):
            

            t_arr+=[t]
            t+=dt
            # step_start_time = time.perf_counter()
            
            global_step +=parallel_env
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob     


            # TRY NOT TO MODIFY: execute the game and log data.
            for i in range(frame_skip):
                next_obs, reward, terminations, truncations, infos = env.step(action.cpu().numpy())

            ## Add this due to jax issue
            reward = np.array(reward)
            next_obs = np.array(next_obs)

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)   
            

            if infos["final_info"]["episode"]["is_final"].sum()>0:

                writer.add_scalar("charts/reward/goal_state", np.mean(np.array(infos["reward_info"]["goal_state"])), global_step)

                writer.add_scalar("charts/episodic_goal_state", np.mean(
                    np.array(infos["reward_info"]["goal_state"][infos["final_info"]["episode"]["is_final"]]
                                )), global_step)

                writer.add_scalar("charts/episodic_return", np.mean(
                    np.array(infos["final_info"]["episode"]["r"][infos["final_info"]["episode"]["is_final"]]
                                )), global_step)
                
                writer.add_scalar("charts/episodic_length", np.mean(
                    np.array(infos["final_info"]["episode"]["l"][infos["final_info"]["episode"]["is_final"]]
                                )), global_step)
                
        print("pre prout")   
        change_obs =  obs.cpu().numpy()  
        xlsindy.render.animate_double_pendulum(link1_length,link2_length,change_obs[:,0,:],np.array(t_arr))
            


        




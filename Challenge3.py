import os
import xml.etree.ElementTree as xml
from os.path import join
from tempfile import TemporaryDirectory
from PIL import Image
import scipy.ndimage

import gymnasium as gym
import matplotlib.pyplot as plt

import imageio
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from tqdm import trange

#TODO: set for cmaes
#from evorob.algorithms.ea_api_sol import EvoAlgAPI
from evorob.algorithms.ea_api import EvoAlgAPI
#from evorob.algorithms.ea_api import CMAES, CMAES_opts
from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import (
    get_distinct_filename,
    get_last_checkpoint_dir,
    get_project_root,
)
from evorob.world.base import World
#from evorob.world.robot.controllers.mlp_sol import NeuralNetworkController
from evorob.world.robot.controllers.mlp import NeuralNetworkController
from evorob.world.robot.controllers.so2 import SO2Controller
from evorob.world.robot.controllers.mlp_hebbian import HebbianController
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

""" 
    Morphology and Controller optimisation: Ant Hill
"""

ROOT_DIR = get_project_root()
ENV_NAME = "AntHill-v0"


class AntWorld(World):

    def __init__(self,):
        action_space = 8  # https://gymnasium.farama.org/environments/mujoco/ant/#action-space
        state_space = 27  # https://gymnasium.farama.org/environments/mujoco/ant/#observation-space

        self.controller = SO2Controller(input_size=state_space,
                                        output_size=action_space,
                                        hidden_size=action_space)

        self.n_weights = self.controller.n_params
        self.n_body_params = 8

        self.n_params = self.n_weights + self.n_body_params
        self.temp_dir = TemporaryDirectory()
        self.world_file = join(self.temp_dir.name, "AntHillEnv.xml")
        self.create_terrain_file("terrain.png")
        self.base_xml_path = join(ROOT_DIR, "evorob", "world", "robot", "assets", "hill_world.xml")

        self.joint_limits = [[-30, 30], [30, 70],
                        [-30, 30], [-70, -30],
                        [-30, 30], [-70, -30],
                        [-30, 30], [30, 70], ]
        self.joint_axis = [[0, 0, 1], [-1, 1, 0],
                      [0, 0, 1], [1, 1, 0],
                      [0, 0, 1], [-1, 1, 0],
                      [0, 0, 1], [1, 1, 0],
                      ]

    def update_robot_xml(self, genotype: np.ndarray):
        points, connectivity_mat = self.geno2pheno(genotype)
        robot = AntRobot(points, connectivity_mat, self.joint_limits, self.joint_axis, verbose=False)
        robot.xml = robot.define_robot()
        robot.write_xml(self.temp_dir.name)

        #% Defining the Robot environment in MuJoCo
        world = xml.parse(self.base_xml_path)
        robot_env = world.getroot()

        robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
        world_xml = xml.tostring(robot_env, encoding="unicode")
        with open(self.world_file, "w") as f:
            f.write(world_xml)

    def create_env(self, render_mode: str = "rgb_array", n_envs: int = 1, max_episode_steps: int = 1000, reset_noise_scale=0.1, **kwargs):
        envs = AsyncVectorEnv(
            [
                lambda i_env=i_env: gym.make(
                    ENV_NAME,
                    robot_path=self.world_file,
                    reset_noise_scale=reset_noise_scale,
                    max_episode_steps=max_episode_steps,
                    render_mode=render_mode,
                )
                for i_env in range(n_envs)
            ]
        )
        return envs

    def geno2pheno(self, genotype):
        #print("geno2pheno sos")
        #to run evol
        #control_weights = genotype[:self.n_weights]*0.1
        #To run challenge 1
        control_weights = genotype[:self.n_weights]*1.0
        body_params = (genotype[0:8]+1)/4+0.1
        #body_params = (genotype[self.n_weights:]+1)/4+0.1
        print(len(body_params))
        print(self.n_body_params)
        assert len(body_params) == self.n_body_params
        assert len(control_weights) == self.n_weights
        assert not np.any(body_params <= 0)

        self.controller.geno2pheno(control_weights)

        front_left_leg, front_left_ankle, front_right_leg, front_right_ankle, back_left_leg, back_left_ankle, back_right_leg, back_right_ankle, = body_params

        # Define the 3D coordinates of the relative tree structure
        front_left_hip_xyz = np.array([0.2, 0.2, 0])
        front_left_knee_xyz = np.array([np.sqrt(0.5 * front_left_leg ** 2), np.sqrt(0.5 * front_left_leg ** 2), 0]) + front_left_hip_xyz
        front_left_toe_xyz = np.array([np.sqrt(0.5 * front_left_ankle ** 2), np.sqrt(0.5 * front_left_ankle ** 2), 0]) + front_left_knee_xyz

        front_right_hip_xyz = np.array([-0.2, 0.2, 0])
        front_right_knee_xyz = np.array([-np.sqrt(0.5 * front_right_leg ** 2), np.sqrt(0.5 * front_right_leg ** 2), 0]) + front_right_hip_xyz
        front_right_toe_xyz = np.array([-np.sqrt(0.5 * front_right_ankle ** 2), np.sqrt(0.5 * front_right_ankle ** 2), 0]) + front_right_knee_xyz

        back_left_hip_xyz = np.array([-0.2, -0.2, 0])
        back_left_knee_xyz = np.array([-np.sqrt(0.5 * back_left_leg ** 2), -np.sqrt(0.5 * back_left_leg ** 2), 0]) + back_left_hip_xyz
        back_left_toe_xyz = np.array([-np.sqrt(0.5 * back_left_ankle ** 2), -np.sqrt(0.5 * back_left_ankle ** 2), 0]) + back_left_knee_xyz

        back_right_hip_xyz = np.array([0.2, -0.2, 0])
        back_right_knee_xyz = np.array([np.sqrt(0.5 * back_right_leg ** 2), -np.sqrt(0.5 * back_right_leg ** 2), 0]) + back_right_hip_xyz
        back_right_toe_xyz = np.array([np.sqrt(0.5 * back_right_ankle ** 2), -np.sqrt(0.5 * back_right_ankle ** 2), 0]) + back_right_knee_xyz

        points = np.vstack([front_left_hip_xyz,
                            front_left_knee_xyz,
                            front_left_toe_xyz,
                            front_right_hip_xyz,
                            front_right_knee_xyz,
                            front_right_toe_xyz,
                            back_left_hip_xyz,
                            back_left_knee_xyz,
                            back_left_toe_xyz,
                            back_right_hip_xyz,
                            back_right_knee_xyz,
                            back_right_toe_xyz,
                            ])

        # define the type of connections [FIXED ARCHITECTURE]
        connectivity_mat = np.array(
            [[150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ]
        )
        return points, connectivity_mat


    def create_terrain_file(self, filename="terrain.png", width=400, depth=400):
        # 1. Create the Slope (Gradient along X)
        # 0.0 at the back, 1.0 at the front
        # TODO: Change the terrain parameters
        slope_deg = 5.0
        bump_scale = 0.1
        sigma = 3.0

        # 1. Create Linear Slope (Gradient along X)
        rise = np.tan(np.deg2rad(slope_deg))
        slope_factor = rise
        x = np.linspace(0, 1, depth)
        y = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)

        # Use tan to get actual height ratio, but clip to 1.0 to stay within hfield Z-bounds
        # (Assuming max height in XML is defined as the Z-scale)
        slope_map = X * slope_factor

        # 2. Add Bumps (Noise)
        rng = np.random.default_rng(42)
        noise = rng.uniform(0, 1, (width, depth))
        gentle_bump = np.tanh(X*10)
        noise = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min())*gentle_bump
        noise_map = noise * bump_scale

        terrain = slope_map + noise_map
        terrain = np.clip(terrain, 0, 1)
        terrain[-1, -1] = 1
        terrain_normalized = (terrain * 255).astype(np.uint8)

        img = Image.fromarray(terrain_normalized, mode='L')
        save_path = os.path.join(self.temp_dir.name, filename)
        img.save(save_path)



    def evaluate_individual(self, genotype, n_repeats=10, n_steps=500):
        self.update_robot_xml(genotype)
        envs = self.create_env(n_envs=n_repeats, max_episode_steps=n_steps)
        self.controller.reset_controller(batch_size=n_repeats)

        rewards_full = np.zeros((n_steps, n_repeats))
        multi_obj_rewards_full = np.zeros((n_steps, n_repeats, 2))

        observations, info = envs.reset()
        done_mask = np.zeros(n_repeats, dtype=bool)
        for step in range(n_steps):
            actions = np.where(done_mask[:, None], 0, self.controller.get_action(observations))
            observations, rewards, dones, truncated, infos = envs.step(actions)

            # Store rewards for active environments only
            # TODO: design appropriate rewards
            forward = np.array(infos.get("reward_forward", np.zeros(n_repeats)))
            healthy = np.array(infos.get("healthy_reward", np.zeros(n_repeats)))
            ctrl    = np.array(infos.get("ctrl_cost",       np.zeros(n_repeats)))

            rewards_full[step, ~done_mask] = (forward + healthy - 0.5 * ctrl)[~done_mask]
            #rewards_full[step, ~done_mask] = rewards[~done_mask]

            # TODO: design appropriate moo-rewards
            multi_obj_reward = np.array([infos["z_velocity"], -infos["ctrl_cost"]]).T # TODO
            multi_obj_rewards_full[step, ~done_mask] = multi_obj_reward[~done_mask]

            obj1 = forward + healthy          # objectif 1 : avancer et rester en vie
            obj2 = -ctrl                      # objectif 2 : efficacité énergétique
            multi_obj_reward = np.stack([obj1, obj2], axis=1)  # shape (n_repeats, 2)
            multi_obj_rewards_full[step, ~done_mask] = multi_obj_reward[~done_mask]

            # Update the done mask based on the "done" and "truncated" flags
            done_mask = done_mask | dones | truncated

            # Optionally, break if all environments have terminated
            if np.all(done_mask):
                break
        final_rewards = np.sum(rewards_full, axis=0)
        final_multi_obj_rewards = np.sum(multi_obj_rewards_full, axis=0)
        envs.close()
        return np.mean(final_rewards), np.mean(final_multi_obj_rewards, axis=0)


# ---------------------------------------------------------------------------
# Evaluation helpers (Challenge 3)
# ---------------------------------------------------------------------------

def _run_episodes_hill(world, genotype, n_episodes, max_episode_steps, seed):
    """Run n_episodes on the hill terrain, returning per-episode stats.

    Returns:
        episode_rewards: total reward per episode
        episode_obj1:    cumulative (reward_forward + healthy_reward) per episode
        episode_obj2:    cumulative (-ctrl_cost) per episode
    """
    world.update_robot_xml(genotype)
    env = gym.make(
        ENV_NAME,
        robot_path=world.world_file,
        max_episode_steps=max_episode_steps,
    )

    rng = np.random.default_rng(seed)
    episode_rewards, episode_obj1, episode_obj2 = [], [], []

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=ep_seed)
        world.controller.reset_controller(batch_size=1)

        total_reward = total_obj1 = total_obj2 = 0.0
        for _ in range(max_episode_steps):
            action = world.controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_obj1 += float(info.get("reward_forward", 0.0)) + float(info.get("healthy_reward", 0.0))
            total_obj2 += -float(info.get("ctrl_cost", 0.0))
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_obj1.append(total_obj1)
        episode_obj2.append(total_obj2)

    env.close()
    return episode_rewards, episode_obj1, episode_obj2


def _record_video_hill(world, genotype, max_steps, seed, out_path):
    """Record one episode to out_path. Returns episode reward or None on failure."""
    try:
        world.update_robot_xml(genotype)
        env = gym.make(
            ENV_NAME,
            robot_path=world.world_file,
            max_episode_steps=max_steps,
            render_mode="rgb_array",
        )
        world.controller.reset_controller(batch_size=1)
        obs, _ = env.reset(seed=seed)
        frames, video_reward = [], 0.0

        for _ in range(max_steps):
            frames.append(env.render())
            action = world.controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, reward, terminated, truncated, _ = env.step(action)
            video_reward += reward
            if terminated or truncated:
                break

        env.close()
        imageio.mimwrite(out_path, frames, fps=20)
        print(f"  Video saved: {out_path}")
        return video_reward
    except Exception as e:
        print(f"  Warning: video skipped ({e})")
        return None


def _stats(values):
    a = np.asarray(values, dtype=float)
    return {
        "values": list(values),
        "mean": float(np.mean(a)),
        "std":  float(np.std(a)),
        "best": float(np.max(a)),
        "worst": float(np.min(a)),
        "median": float(np.median(a)),
    }


def evaluate_checkpoint(
    checkpoint_dir: str,
    output_dir: str = "evaluation_output",
    n_episodes: int = 256,          # set to 256 for submission; lower for testing
):
    """Evaluate a Challenge-3 NSGA-II checkpoint on the hilly terrain.

    Identifies two specialists (best per objective) and a generalist (best
    combined score) from the last checkpoint's population, evaluates each for
    n_episodes, records three videos, and writes a score file.

    The controller type and genotype size are taken from the AntWorld default
    (whatever the student configured), so no hardcoded assumptions are made.
    Objectives: [reward_forward + healthy_reward,  -ctrl_cost].

    Args:
        checkpoint_dir: Path to your NSGA-II checkpoint folder.
        output_dir:     Where to save the score file and videos.
        n_episodes:     Episodes per individual (256 for submission).
    """
    max_episode_steps: int = 1000  # DO NOT CHANGE!
    seed: int = 0                  # DO NOT CHANGE!

    # --- Locate checkpoint files ---
    last_gen = get_last_checkpoint_dir(checkpoint_dir)

    def _try_load(filename):
        for d in ([last_gen] if last_gen else []) + [checkpoint_dir]:
            p = os.path.join(d, filename)
            if os.path.isfile(p):
                return np.load(p, allow_pickle=True)
        return None

    x_best = _try_load("x_best.npy")
    if x_best is None:
        print(f"ERROR: Could not find x_best.npy in '{checkpoint_dir}'.")
        return None

    population = _try_load("x.npy")
    fitness    = _try_load("f.npy")
    print(f"Loaded x_best  (shape: {x_best.shape})")

    # --- Identify specialist and generalist genotypes ---
    if (population is not None and fitness is not None
            and fitness.ndim == 2 and fitness.shape[1] >= 2):
        spec1_idx = int(np.argmax(fitness[:, 0]))           # best forward+healthy
        spec2_idx = int(np.argmax(fitness[:, 1]))           # best efficiency
        gen_idx   = int(np.argmax(np.sum(fitness, axis=1))) # pareto-knee proxy
        spec1_g, spec2_g, gen_g = population[spec1_idx], population[spec2_idx], population[gen_idx]
        print(f"Specialist obj1 (forward): idx={spec1_idx}  f={fitness[spec1_idx]}")
        print(f"Specialist obj2 (effic.) : idx={spec2_idx}  f={fitness[spec2_idx]}")
        print(f"Generalist (best sum)    : idx={gen_idx}    f={fitness[gen_idx]}")
    else:
        print("Warning: population/fitness not found — using x_best for all three roles.")
        spec1_g = spec2_g = gen_g = x_best
    # --- Comparaison des 8 premières valeurs des génotypes ---
    print("\nComparaison des 8 premières valeurs (génotypes spécialistes) :")
    print("Index | Specialist obj1 | Specialist obj2")
    print("------------------------------------------")

    g1 = spec1_g[:8]
    g2 = spec2_g[:8]

    for i in range(8):
        print(f"{i:5d} | {g1[i]:16.6f} | {g2[i]:16.6f}")

    # --- AntWorld uses whatever controller the student configured ---
    world = AntWorld()
    controller_name = type(world.controller).__name__
    print(f"Controller: {controller_name}  |  params={world.controller.n_params}  |  genotype size={world.n_params}\n")

    individuals = {
        "specialist_obj1": spec1_g,
        "specialist_obj2": spec2_g,
        "generalist":      gen_g,
    }

    # --- Run episodes ---
    results = {}
    for label, genotype in individuals.items():
        print(f"Evaluating {label} ({n_episodes} episodes)...")
        rewards, obj1_vals, obj2_vals = _run_episodes_hill(
            world, genotype, n_episodes, max_episode_steps, seed
        )
        results[label] = {
            "reward": _stats(rewards),
            "obj1":   _stats(obj1_vals),
            "obj2":   _stats(obj2_vals),
        }
        r = results[label]
        print(f"  reward: {r['reward']['mean']:.2f} +/- {r['reward']['std']:.2f}  "
              f"obj1: {r['obj1']['mean']:.2f}  obj2: {r['obj2']['mean']:.2f}")

    # --- Record videos ---
    os.makedirs(output_dir, exist_ok=True)
    video_names = {
        "specialist_obj1": "specialist_forward",
        "specialist_obj2": "specialist_efficiency",
        "generalist":      "generalist",
    }
    for label, genotype in individuals.items():
        vpath = os.path.join(output_dir, f"evaluation_{video_names[label]}.mp4")
        _record_video_hill(world, genotype, max_episode_steps, seed, vpath)

    # --- Score file ---
    score_path = os.path.join(output_dir, "evaluation_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Challenge 3 - Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Controller      : {controller_name} ({world.controller.n_params} params)\n")
        f.write(f"Genotype size   : {world.n_params}  (weights={world.controller.n_params}, body={world.n_body_params})\n")
        f.write(f"Checkpoint      : {checkpoint_dir}\n")
        f.write(f"Episodes/indiv. : {n_episodes}\n")
        f.write(f"Objectives      : [reward_forward+healthy_reward, -ctrl_cost]\n\n")

        f.write("=" * 72 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 72 + "\n")
        f.write(f"{'Individual':<22s} {'Rew.Mean':>9s} {'Rew.Std':>8s} {'Rew.Best':>9s} "
                f"{'Obj1.Mean':>10s} {'Obj2.Mean':>10s}\n")
        f.write("-" * 72 + "\n")
        for label in individuals:
            r = results[label]
            f.write(f"{label:<22s} {r['reward']['mean']:9.2f} {r['reward']['std']:8.2f} "
                    f"{r['reward']['best']:9.2f} {r['obj1']['mean']:10.2f} {r['obj2']['mean']:10.2f}\n")
        f.write("\n")

        for label in individuals:
            r = results[label]
            f.write("-" * 50 + "\n")
            f.write(f"{label.upper()} — Per-episode rewards\n")
            f.write("-" * 50 + "\n")
            for i, rew in enumerate(r["reward"]["values"]):
                f.write(f"  Episode {i + 1:3d}: {rew:10.2f}\n")
            f.write("\n")

    print(f"\nScore saved to: {score_path}")
    print("=" * 60)
    for label in individuals:
        r = results[label]
        print(f"  {label:<22s}: reward={r['reward']['mean']:.2f} +/-{r['reward']['std']:.2f}"
              f"  obj1={r['obj1']['mean']:.2f}  obj2={r['obj2']['mean']:.2f}")
    print("=" * 60)
    return results


def run_EA_single(ea_single, world):
    print("SINGLE")
    for _ in trange(ea_single.n_gen):
        pop = ea_single.ask()
        fitnesses_gen = np.empty(len(pop))
        for index, genotype in enumerate(pop):
            fit_ind, _ = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea_single.tell(pop, fitnesses_gen, save_checkpoint=True)


def run_EA_multi(ea_multi, world):
    print("Multi")
    for _ in trange(ea_multi.n_gen):
        pop = ea_multi.ask()
        fitnesses_gen = np.empty((len(pop), 2))
        for index, genotype in enumerate(pop):
            _, fit_ind = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea_multi.tell(pop, fitnesses_gen, save_checkpoint=True)

def plot_pareto_fronts_from_checkpoint(checkpoint_dir: str):
    """
    Loads fitness data from a checkpoint directory and plots Pareto fronts using NSGA-II sorting.
    """
    # Load all generations' fitness data
    fitness_path = f"{checkpoint_dir}/f.npy"
    try:
        all_fitness = np.load(fitness_path)
    except Exception as e:
        print(f"Could not load fitness data from {fitness_path}: {e}")
        return

    # If only 2D objectives, shape is (generations, pop, 2)
    if all_fitness.ndim == 3:
        # Plot only the last generation by default
        fitness = all_fitness[-1]
    else:
        fitness = all_fitness

    nsga = NSGAII(
        population_size=fitness.shape[0], n_opt_params=560
    )  # n_opt_params is a dummy here
    fronts, _ = nsga.fast_nondominated_sort(fitness)

    plt.figure(figsize=(8, 6))
    n_fronts = len(fronts)
    colors = plt.cm.viridis(np.linspace(0, 1, n_fronts))
    for i, front in enumerate(fronts[:n_fronts]):
        front_fitness = fitness[front]
        plt.scatter(
            front_fitness[:, 0],
            front_fitness[:, 1],
            label=f"Front {i}",
            color=colors[i],
            alpha=0.7,
        )

    plt.xlabel("Fitness Objective 1")
    plt.ylabel("Fitness Objective 2")
    plt.title("Pareto Fronts (Last Generation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pareto_fronts.png")
    plt.show()
    print("Pareto fronts plotted and saved as pareto_fronts.png")


def main():
    #%% Optimise single-objective
    CMAES_opts = {
    "min": -1,
    "max": 1,
    "mutation_sigma": 0.3,
    "num_generations": 100
    }
    world = AntWorld()
    n_parameters = world.n_params

    #%% Understanding the world
    """
    genotype = np.random.uniform(-1, 1, n_parameters)
    world.update_robot_xml(genotype)
    world.visualise_individual(genotype)
    print("RUN MAIN")
    """
    # TODO Overwrite controller and load best run exercise 1
    """
    world = AntWorld()
    state_space = 27
    action_space = 8 # Change controller
    world.controller = NeuralNetworkController(input_size=state_space,
                                               output_size=action_space,
                                               hidden_size=16)
    world.n_weights = world.controller.n_params
    world.n_params = world.n_weights + world.n_body_params
    n_parameters = world.n_params
    genotype = np.random.uniform(-1, 1, n_parameters)
    print("RUN MAIN")

    result_dir = ...
    prev_best = np.load("C:/Users/hmpau/Documents/MA2/Evolutionary Robotics/Exercise/micro-515-EvoRob/Result Challenge1/x_best.npy") # load previous run
    print(prev_best)
    #print("prev_best:", len(prev_best))
    #print("n_weights:", world.n_weights)
    genotype[:-8] = prev_best

    genotype[-8::2] = -0.6  # fix upper leg length 0.2
    genotype[-7::2] = 1  # fix lower leg length 0.6
    world.update_robot_xml(genotype)
    #world.geno2pheno(genotype)          # ← initialise le contrôleur
    world.visualise_individual(genotype)
    """
    #%% Evolve open-loop so2
    """
    world = AntWorld()
    world.n_weights = world.controller.n_params
    world.n_params = world.n_weights + world.n_body_params
    n_parameters = world.n_params
    population_size = 50   #150
    opts = CMAES_opts.copy()
    opts["min"] = -1
    opts["max"] = 1
    opts["mutation_sigma"] = 0.3
    opts["num_generations"] = 25  #50

    results_dir = join(ROOT_DIR, "results", ENV_NAME, "single")
    ea_single = EvoAlgAPI(
    n_params=n_parameters,
    population_size=population_size,
    num_generations=opts["num_generations"],
    output_dir=results_dir,
    sigma=0.3
    )

    run_EA_single(ea_single, world)
    
    #%% visualise
    checkpoint = get_last_checkpoint_dir(results_dir)
    best_individual = np.load(join(results_dir, checkpoint, "x_best.npy"))
    world.update_robot_xml(best_individual)
    env = world.create_env(max_episode_steps=-1)
    video_name = get_distinct_filename(join(results_dir, "best.mp4"))
    print(f"Finished ES run, generating video [{video_name}]...")
    world.generate_best_individual_video(env, video_name=video_name, n_steps=500)
    """

    #%% Optimise multi-objective
    
    world = AntWorld()
    state_space = 27
    action_space = 8 # Change controller
    world.controller = NeuralNetworkController(input_size=state_space,
                                               output_size=action_space,
                                               hidden_size=16)
    world.n_weights = world.controller.n_params
    world.n_params = world.n_weights + world.n_body_params
    n_parameters = world.n_params
    print("Number of parameters:", n_parameters)
    print("Number of weights:", world.n_weights)
    population_size = 50   #100 par défaut

    opts = {}
    opts["min"] = -1
    opts["max"] = 1
    opts["num_parents"] = population_size//2
    opts["num_generations"] = 50
    opts["mutation_prob"] = 0.2
    opts["crossover_prob"] = 0.5

    results_dir = join(ROOT_DIR, "results", ENV_NAME, "multi_2")
    ea_multi_obj = NSGAII(
        population_size=population_size,
        n_opt_params=n_parameters,
        n_parents= opts["num_parents"],
        bounds=(opts["min"], opts["max"]),
        num_generations=opts["num_generations"],
        mutation_prob=opts["mutation_prob"],
        crossover_prob=opts["crossover_prob"],
        output_dir=results_dir,
        #pretrained_path= "Result_ch2/best_flat_controller.npy",
        noise_std=0.15,
    )




    """
    ea_multi_obj = NSGAII(population_size,
                          n_parameters,
                          opts["num_parents"],
                          opts["num_generations"],
                          (opts["min"], opts["max"]),
                          opts["mutation_prob"],
                          opts["crossover_prob"])
    ea_multi_obj.directory_name = results_dir
    """
    #run_EA_multi(ea_multi_obj, world)

    #%% visualise
    checkpoint = get_last_checkpoint_dir(results_dir)
    """
    best_individual = np.load(join(results_dir, checkpoint, "x_best.npy"), allow_pickle=True)
    #best_individual = np.load("results/AntHill-v0/multi/46/x_best.npy")
    world.update_robot_xml(best_individual)
    env = world.create_env(max_episode_steps=-1)
    video_name = get_distinct_filename(join(results_dir, "best.mp4"))
    print(f"Finished NSGAII run, generating video [{video_name}]...")
    world.generate_best_individual_video(env, video_name=video_name, n_steps=500)
    """
    evaluate_checkpoint("c:/Users/hmpau/Documents/MA2/Evolutionary Robotics/Exercise/micro-515-EvoRob/results/AntHill-v0/multi_2")

     # Uncomment to plot Pareto fronts from checkpoint
    #plot_pareto_fronts_from_checkpoint(
    #    checkpoint_dir="c:/Users/hmpau/Documents/MA2/Evolutionary Robotics/Exercise/micro-515-EvoRob/results/AntHill-v0/multi_2/49"
    #)
if __name__ == "__main__":
    main()

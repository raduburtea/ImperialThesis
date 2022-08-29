import sys
from venv import create
sys.path.append('./or-gym')
sys.path.append('./rllab/')

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

import or_gym
from or_gym.utils import create_env
from or_gym.envs.supply_chain.inventory_management import InvManagementMasterEnv
import pickle

import neptune.new as neptune


params = {"use_neptune": False,
        "max_path_len": 30,
         "Inv": [100, 100, 200],
         "Capacity": [100, 90, 80],
         'n_itr': 150,
          "discount": 0.99,
        'batch_size': 39000,
         }

if params['use_neptune']:
        #Please include your neptune.ai credentials
        neptune_instance = neptune.init(
            project="sample_name",
            api_token="sample_token",
        )  # your credentials

        neptune_instance['params'].log(params)
else:
    neptune_instance = None


def run_task(*_):
    # Please note that different environments with different action spaces may require different
    # policies. For example with a Box action space, a GaussianMLPPolicy works, but for a Discrete
    # action space may need to use a CategoricalMLPPolicy (see the trpo_gym_cartpole.py example)
    env = InvManagementMasterEnv(neptune_instance, max_path_length = params['max_path_len'], inventory = params['Inv'], capacity=params['Capacity'])

    policy = GaussianMLPPolicy(
        env_spec=env,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64,32)
    )

    baseline = LinearFeatureBaseline(env_spec=env)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=params['batch_size'],
        max_path_length=params['max_path_len'],
        n_itr=params['n_itr'],
        discount=params['discount'],
        step_size=1,
        neptune_instance = neptune_instance
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()

run_task()



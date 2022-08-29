import sys
sys.path.append('./or-gym')
sys.path.append('./rllab')
import numpy as np
from rllab.misc.instrument import run_experiment_lite
from or_gym.envs.supply_chain.inventory_management import InvManagementMasterEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.cpo.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.cpo.algos.safe.cpo import CPO
from sandbox.cpo.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.cpo.safety_constraints.inventory_management_safe import InvManagmentSafetyConstraint
import lasagne.nonlinearities as NL
import neptune.new as neptune
import warnings
warnings.filterwarnings("ignore")

params = {
        "network_size": (64,32),
        "max_path_length": 30,
        "n_itr": 200,
        "gae_lambda": 0.95,
        "discount": 0.99,
        'inventory': [100, 100, 200],
        'capacity': [100, 90, 80],
        'batch_size': 39000,
        'max_value': 0.0001,
        'use_neptune': False,
    }

if params['use_neptune']:
        #Please include your neptune.ai credentials
        run = neptune.init(
            project="sample_name",
            api_token="sample_token",
        )  # your credentials

        run['params'].log(params)
else:
    run = None


def run_task(run):

        trpo_stepsize = 1
        trpo_subsample_factor = 2

        #Initialize the environment with the desired parameters
        env = InvManagementMasterEnv(run, max_path_length = params['max_path_length'], inventory = params['inventory'], \
            capacity=params['capacity'], flag = False)

        #Initialize policy network
        policy = GaussianMLPPolicy(env,
                    hidden_sizes=(64,32)
                 )

        #Initialize baseline
        baseline = GaussianMLPBaseline(
            env_spec=env,
            regressor_args={
                    'hidden_sizes': (64,32),
                    'hidden_nonlinearity': NL.tanh,
                    'learn_std':False,
                    'step_size':trpo_stepsize,
                    'optimizer':ConjugateGradientOptimizer()
                    }
        )

        #Initialize safety baseline 
        safety_baseline = GaussianMLPBaseline(
            env_spec=env,
            regressor_args={
                    'hidden_sizes': (64,32),
                    'hidden_nonlinearity': NL.tanh,
                    'learn_std':False,
                    'step_size':trpo_stepsize,
                    'optimizer':ConjugateGradientOptimizer()
                    },
            target_key='safety_rewards',
            )

        #Initialize the safety constraint, used for constraint evaluation
        safety_constraint = InvManagmentSafetyConstraint(env, max_value=params['max_value'], baseline=safety_baseline)

        #Initialize the CPO algorithm with the given parameters
        algo = CPO(
            env=env,
            policy=policy,
            baseline=baseline,
            safety_constraint=safety_constraint,
            safety_gae_lambda=1,
            batch_size=params['batch_size'],
            max_path_length=params['max_path_length'],
            n_itr=params["n_itr"],
            gae_lambda=params["gae_lambda"],
            discount=params["discount"],
            step_size=trpo_stepsize,
            optimizer_args={'subsample_factor':trpo_subsample_factor},
            neptune_instance = run,
        )
        #Call the trainer and begin training the agent 
        algo.train()

#Run the experiment in a distributed manner
run_experiment_lite(
    run_task(run),
    n_parallel=4,
    snapshot_mode="last",
    exp_prefix='CPO-InvManagement',
    seed=20,
)

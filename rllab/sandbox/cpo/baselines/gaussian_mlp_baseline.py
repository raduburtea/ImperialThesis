import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor


class GaussianMLPBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            num_seq_inputs=1,
            regressor_args=None,
            target_key=None,
    ):
        Serializable.quick_init(self, locals())
        super(GaussianMLPBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        if target_key == None:
            target_key = 'rewards'
            
        self._regressor = GaussianMLPRegressor(
            input_shape=(env_spec.observation_space.flat_dim * num_seq_inputs,),
            output_dim=1,
            name='vf_'+target_key,
            **regressor_args
            )

        self._regressor_safety = GaussianMLPRegressor(
            input_shape=(env_spec.observation_space.flat_dim * num_seq_inputs,),
            output_dim=1,
            name='vf_'+target_key,
            **regressor_args
            )
        self._target_key = target_key
        

    @overrides
    def fit(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p[self._target_key] for p in paths])
        if self._target_key == "safety_rewards":
            self._regressor_safety.fit(observations, returns.reshape((-1, 1)))
        else:
            self._regressor.fit(observations, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path):
        if self._target_key == "safety_rewards":
            return self._regressor_safety.predict(path["observations"]).flatten()
        return self._regressor.predict(path["observations"]).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)

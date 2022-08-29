

import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger


class Sampler(object):
    def start_worker(self):
        """
        Initialize the sampler, e.g. launching parallel workers if necessary.
        """
        raise NotImplementedError

    def obtain_samples(self, itr):
        """
        Collect samples for the given iteration number.
        :param itr: Iteration number.
        :return: A list of paths.
        """
        raise NotImplementedError

    def process_samples(self, itr, paths):
        """
        Return processed sample data (typically a dictionary of concatenated tensors) based on the collected paths.
        :param itr: Iteration number.
        :param paths: A list of collected paths.
        :return: Processed sample data.
        """
        raise NotImplementedError

    def shutdown_worker(self):
        """
        Terminate workers if necessary.
        """
        raise NotImplementedError


class BaseSampler(Sampler):
    def __init__(self, algo, neptune_instance):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo
        self.neptune_instance = neptune_instance

    def process_samples(self, itr, paths):
        baselines = []
        returns = []

        if hasattr(self.algo.baseline, "predict_n"):
            all_path_baselines = self.algo.baseline.predict_n(paths)
        else:
            all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )
        penalties = [path["env_infos"]['Penalty'] for path in paths]

        node1_ro = [np.mean(path['env_infos']['Node1/Replenishment order'][:-10]) for path in paths]
        node2_ro = [np.mean(path['env_infos']['Node2/Replenishment order'][:-10]) for path in paths]
        node3_ro = [np.mean(path['env_infos']['Node3/Replenishment order'][:-10]) for path in paths]
        node1_ic = [np.mean(path['env_infos']['Node1/Inventory constraint'][:-10]) for path in paths]
        node2_ic = [np.mean(path['env_infos']['Node2/Inventory constraint'][:-10]) for path in paths]
        node1_cc = [np.mean(path['env_infos']['Node1/Capacity constraint'][:-10]) for path in paths]
        node2_cc = [np.mean(path['env_infos']['Node2/Capacity constraint'][:-10]) for path in paths]
        node3_cc = [np.mean(path['env_infos']['Node3/Capacity constraint'][:-10]) for path in paths]


        node1_ro_max = [np.max(path['env_infos']['Node1/Replenishment order']) for path in paths]
        node2_ro_max = [np.max(path['env_infos']['Node2/Replenishment order']) for path in paths]
        node3_ro_max = [np.max(path['env_infos']['Node3/Replenishment order']) for path in paths]

        if not self.algo.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns=returns,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = tensor_utils.pad_tensor_n(obs, max_path_length)

            if self.algo.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = tensor_utils.pad_tensor_n(actions, max_path_length)

            rewards = [path["rewards"] for path in paths]
            rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

            returns = [path["returns"] for path in paths]
            returns = tensor_utils.pad_tensor_n(returns, max_path_length)

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = tensor_utils.pad_tensor_n(valids, max_path_length)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.sum(self.algo.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        if self.neptune_instance is not None:
            self.neptune_instance["Penalty"].log(np.mean(penalties))
            self.neptune_instance["MaxReturn"].log(np.max(undiscounted_returns))
            self.neptune_instance["MinReturn"].log(np.min(undiscounted_returns))        
            self.neptune_instance["StdReturn"].log(np.std(undiscounted_returns))
            self.neptune_instance["AverageDiscountedReturn"].log(average_discounted_return)       
            self.neptune_instance["AverageReturn"].log(np.mean(undiscounted_returns))
            self.neptune_instance["Iteration"].log(itr)
            self.neptune_instance['Node1/Replenishment order'].log(np.mean(node1_ro))
            self.neptune_instance['Node2/Replenishment order'].log(np.mean(node2_ro))
            self.neptune_instance['Node3/Replenishment order node 3'].log(np.mean(node3_ro))
            self.neptune_instance['Node1/Inventory constraint'].log(np.mean(node1_ic))
            self.neptune_instance['Node2/Inventory constraint'].log(np.mean(node2_ic))
            self.neptune_instance['Node1/Capacity constraint'].log(np.mean(node1_cc))
            self.neptune_instance['Node2/Capacity constraint'].log(np.mean(node2_cc))
            self.neptune_instance['Node3/Capacity constraint'].log(np.mean(node3_cc))
            self.neptune_instance['Node1/RO MAX'].log(np.mean(node1_ro_max))
            self.neptune_instance['Node2/RO MAX'].log(np.mean(node2_ro_max))
            self.neptune_instance['Node3/RO MAX'].log(np.mean(node3_ro_max))
        logger.log("fitting baseline...")
        if hasattr(self.algo.baseline, 'fit_with_samples'):
            self.algo.baseline.fit_with_samples(paths, samples_data)
        else:
            self.algo.baseline.fit(paths)
        logger.log("fitted")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        # raise Exception('called')
        print('Ihaaaaaaa:   ', np.max(undiscounted_returns))
        print('Magari:   ', np.std(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data

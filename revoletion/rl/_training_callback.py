import collections
import logging

from stable_baselines3.common.callbacks import BaseCallback

from .environment import (
    INFO_KEY_REWARD_COMPONENTS,
    INFO_KEY_STATUS,
    EnvironmentStepStatus,
    RewardComponents,
)

_LOGGER = logging.getLogger(__name__)

_BASE_TRACE_KEY = "revoletion"
_TRACE_KEY_DONE_COUNT = "done_count"
_TRACE_KEY_INFEASIBILITY_COUNT = "infeasibility_count"
_TRACE_KEY_INFEASIBILITY_RATE = "infeasibility_rate"
_TRACE_KEY_INFEASIBILITY = "infeasibility_reward"
_TRACE_KEY_REWARD = "mean_step_reward"
_TRACE_KEY_GRID_OPEX_REWARD = "grid_opex_reward"
_TRACE_KEY_GEN_COST = "gen_opex_reward"
_TRACE_KEY_CHARGE_COST = "charge_opex_reward"
_TRACE_KEY_EXT_CHARGE_COST = "ext_charge_opex_reward"
_TRACE_KEY_SOC_DIFF_REWARD = "soc_diff_reward"
_TRACE_KEY_SOC_VIOLATIONS_COUNT = "soc_violations_count"
_TRACE_KEY_SOC_VIOLATIONS_RATE = "soc_violations_rate"
_TRACE_KEY_SOC_VIOLATIONS_MEAN = "soc_violations_mean"
_TRACE_KEY_POWER_DIFF_REWARD = "power_diff_reward"
_TRACE_KEY_ATBASE_VIOLATION_REWARD = "atbase_violation_reward"

_ENABLED_TRACES = [
    _TRACE_KEY_REWARD,
    _TRACE_KEY_GRID_OPEX_REWARD,
    _TRACE_KEY_SOC_DIFF_REWARD,
    _TRACE_KEY_SOC_VIOLATIONS_RATE,
    _TRACE_KEY_SOC_VIOLATIONS_MEAN,
    _TRACE_KEY_POWER_DIFF_REWARD,
    _TRACE_KEY_ATBASE_VIOLATION_REWARD,
    _TRACE_KEY_INFEASIBILITY_COUNT,
]


class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0, stats_window_size: int = 100):
        super().__init__(verbose)

        self._stats_window_size = stats_window_size

        self._infeasible_count = 0

        self._soc_count = 0
        self._soc_violation_count = 0

        self._queues = {}
        for trace_key in _ENABLED_TRACES:
            self._queues[trace_key] = collections.deque(maxlen=self._stats_window_size)

        self._trace_keys = {}
        trace_idx = 0
        for key in _ENABLED_TRACES:
            trace_key = f"{_BASE_TRACE_KEY}/{trace_idx:02d}_{key}"
            self._trace_keys[key] = trace_key
            trace_idx += 1

    def _on_step(self) -> bool:
        vec_infos = self.locals["infos"]
        vec_dones = self.locals["dones"]
        for infos, done in zip(vec_infos, vec_dones):
            reward: RewardComponents = infos[INFO_KEY_REWARD_COMPONENTS]

            self._record_mean_trace(_TRACE_KEY_REWARD, reward.total_reward)
            self._record_mean_trace(_TRACE_KEY_GRID_OPEX_REWARD, reward.grid_opex_reward)
            # self._record_mean_trace(_TRACE_KEY_GEN_COST, reward.gen_opex_reward)
            self._record_mean_trace(_TRACE_KEY_SOC_DIFF_REWARD, reward.soc_diff_reward)
            self._record_mean_trace(_TRACE_KEY_POWER_DIFF_REWARD, reward.power_diff_reward)
            self._record_mean_trace(_TRACE_KEY_ATBASE_VIOLATION_REWARD, reward.atbase_violation_reward)

            for soc_diff in reward.soc_diffs:
                self._soc_count += 1
                if soc_diff < 0.0:
                    self._record_mean_trace(_TRACE_KEY_SOC_VIOLATIONS_MEAN, soc_diff)
                    self._record_mean_trace(_TRACE_KEY_SOC_VIOLATIONS_RATE, 1.0)
                else:
                    self._record_mean_trace(_TRACE_KEY_SOC_VIOLATIONS_RATE, 0.0)

            if not done:
                continue

            # self._done_count += 1

            if infos[INFO_KEY_STATUS] == EnvironmentStepStatus.INFEASIBLE:
                # self._status.append(1)
                self._infeasible_count += 1
                self.logger.record(_TRACE_KEY_INFEASIBILITY_COUNT, self._infeasible_count)
            #     self._infeasibility.append(reward.infeasibility_reward)
            #     self.logger.record(_TRACE_KEY_INFEASIBILITY, sum(self._infeasibility) / len(self._infeasibility))
            # else:
            #     self._status.append(0)

            # self.logger.record(_TRACE_KEY_DONE_COUNT, self._done_count)
            # self.logger.record(_TRACE_KEY_INFEASIBILITY_RATE, sum(self._status) / len(self._status))

        return True

    def _record_mean_trace(self, key: str, value: int | float) -> None:
        queue = self._queues[key]
        queue.append(value)

        new_record = sum(queue) / len(queue)
        trace_key = self._trace_keys[key]
        self.logger.record(trace_key, new_record)

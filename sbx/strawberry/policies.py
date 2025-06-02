from sbx.common.type_aliases import RLBatchNormTrainState
from sbx.crossq.policies import CrossQPolicy


class StrawberryPolicy(CrossQPolicy):
    def build(self, key, lr_schedule, qf_learning_rate):
        key = super().build(key, lr_schedule, qf_learning_rate)
        # Convert qf_state to RLBatchNormTrainState with target params
        self.qf_state = RLBatchNormTrainState.create(
            apply_fn=self.qf_state.apply_fn,
            params=self.qf_state.params,
            batch_stats=self.qf_state.batch_stats,
            target_params=self.qf_state.params,
            tx=self.qf_state.tx,
        )
        return key

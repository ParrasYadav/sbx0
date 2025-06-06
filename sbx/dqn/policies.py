from typing import Any, Callable, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.policies import BaseJaxPolicy, Flatten
from sbx.common.type_aliases import RLTrainState


class QNetwork(nn.Module):
    n_actions: int
    n_units: int = 256
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_actions)(x)
        return x


# Add CNN policy from DQN paper
class NatureCNN(nn.Module):
    n_actions: int
    n_units: int = 512
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Convert from channel-first (PyTorch) to channel-last (Jax)
        x = jnp.transpose(x, (0, 2, 3, 1))
        # Convert to float and normalize the image
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = self.activation_fn(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = self.activation_fn(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = self.activation_fn(x)
        # Flatten
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_actions)(x)
        return x


class DQNPolicy(BaseJaxPolicy):
    action_space: spaces.Discrete  # type: ignore[assignment]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        if net_arch is not None:
            assert isinstance(net_arch, list)
            self.n_units = net_arch[0]
        else:
            self.n_units = 256
        self.activation_fn = activation_fn

    def build(self, key: jax.Array, lr_schedule: Schedule) -> jax.Array:
        key, qf_key = jax.random.split(key, 2)

        obs = jnp.array([self.observation_space.sample()])

        self.qf: nn.Module = QNetwork(
            n_actions=int(self.action_space.n),
            n_units=self.n_units,
            activation_fn=self.activation_fn,
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init({"params": qf_key}, obs),
            target_params=self.qf.init({"params": qf_key}, obs),
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.qf.apply = jax.jit(self.qf.apply)  # type: ignore[method-assign]

        return key

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    @staticmethod
    @jax.jit
    def select_action(qf_state, observations):
        qf_values = qf_state.apply_fn(qf_state.params, observations)
        return jnp.argmax(qf_values, axis=1).reshape(-1)

    def _predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:  # type: ignore[override]
        return DQNPolicy.select_action(self.qf_state, observation)


class CNNPolicy(DQNPolicy):
    def build(self, key: jax.Array, lr_schedule: Schedule) -> jax.Array:
        key, qf_key = jax.random.split(key, 2)

        obs = jnp.array([self.observation_space.sample()])

        self.qf = NatureCNN(
            n_actions=int(self.action_space.n),
            n_units=self.n_units,
            activation_fn=self.activation_fn,
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init({"params": qf_key}, obs),
            target_params=self.qf.init({"params": qf_key}, obs),
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )
        self.qf.apply = jax.jit(self.qf.apply)  # type: ignore[method-assign]

        return key

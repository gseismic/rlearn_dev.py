import numpy as np
from gymnasium import spaces

from rlearn_dev.methods.ppo.naive.network.continous import ActorCritic


def test_actor_critic_disables_action_scaling_for_infinite_bounds():
    action_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(2,),
        dtype=np.float32,
    )

    network = ActorCritic((3,), action_space)

    assert network.scale_action is False
    assert not hasattr(network, "action_scale")
    assert not hasattr(network, "action_bias")


def test_actor_critic_keeps_action_scaling_for_finite_bounds():
    action_space = spaces.Box(
        low=np.array([-1.0, -2.0], dtype=np.float32),
        high=np.array([1.0, 2.0], dtype=np.float32),
        dtype=np.float32,
    )

    network = ActorCritic((3,), action_space)

    assert network.scale_action is True
    assert hasattr(network, "action_scale")
    assert hasattr(network, "action_bias")

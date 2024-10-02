import pytest
from rlearn_dev.utils.epsilon_scheduler import EpsilonScheduler

@pytest.fixture
def scheduler():
    return EpsilonScheduler(1.0, 0.1, scheduler_type='linear', total_steps=10)

def test_linear_decay(scheduler):
    for _ in range(10):
        scheduler.step()
        epsilon = scheduler.get_epsilon()
        assert 0.1 <= epsilon <= 1.0

def test_exponential_decay():
    scheduler = EpsilonScheduler(1.0, 0.1, scheduler_type='exponential', decay_rate=0.5)
    for _ in range(10):
        scheduler.step()
        epsilon = scheduler.get_epsilon()
        assert epsilon <= 1.0
        assert epsilon >= 0.1


if __name__ == '__main__':
    pytest.main()
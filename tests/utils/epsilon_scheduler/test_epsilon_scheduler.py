import pytest
from rlearn_dev.utils.epsilon_scheduler import EpsilonScheduler
from matplotlib import pyplot as plt

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

def get_step_epsilons(scheduler, steps): 
    epsilons = []
    for _ in range(steps):
        scheduler.step()
        epsilon = scheduler.get_epsilon()
        epsilons.append(epsilon)
    return epsilons

def plot_linear_cycle_decay():
    scheduler = EpsilonScheduler(1.0, 0.1, scheduler_type='linear_cycle',
                                 total_steps=100, cycle_epsilon=0.1, 
                                 cycle_steps=50)
    epsilons = get_step_epsilons(scheduler, 100)
    plt.plot(epsilons)
    plt.show()

def plot_half_life_cycle_decay():
    scheduler = EpsilonScheduler(1.0, 0.1, 
                                 scheduler_type='half_life_cycle',
                                 half_life=50, cycle_epsilon=0.1, 
                                 cycle_steps=50)
    epsilons = get_step_epsilons(scheduler, 100)
    plt.plot(epsilons)
    plt.show()

if __name__ == '__main__':
    if 0:
        plot_linear_cycle_decay()
    if 1:
        plot_half_life_cycle_decay()
    if 0:
        pytest.main()

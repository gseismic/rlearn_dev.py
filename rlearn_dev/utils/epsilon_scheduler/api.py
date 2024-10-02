from .epsilon_scheduler import EpsilonScheduler

def get_scheduler(scheduler_type, **kwargs):
    epsilon_start = kwargs.pop('epsilon_start', 0.1)    
    epsilon_end = kwargs.pop('epsilon_end', 0.01)
    return EpsilonScheduler(epsilon_start, epsilon_end, 
                            scheduler_type, **kwargs)

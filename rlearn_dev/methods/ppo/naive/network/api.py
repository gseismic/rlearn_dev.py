from .discrete import MLPActor, MLPCritic


def get_actor_model(env, model_type='MLPActor', model_kwargs={}):
    assert model_type == 'MLPActor'
    actor = MLPActor(env)
    return actor

def get_critic_model(env, model_type='MLPCritic', model_kwargs={}):
    assert model_type == 'MLPCritic'
    critic = MLPCritic(env)
    return critic
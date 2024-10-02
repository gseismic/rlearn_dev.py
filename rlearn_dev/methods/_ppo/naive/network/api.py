from .network import ActorCriticMLP

def get_model(state_dim, action_dim, model_type='ActorCriticMLP', model_kwargs=None):
    if model_type == 'ActorCriticMLP':
        return ActorCriticMLP(state_dim, action_dim, **(model_kwargs or {}))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

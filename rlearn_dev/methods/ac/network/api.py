from .network import IndependentActorCriticMLP


def get_model(state_dim, action_dim, model_type, model_kwargs={}):
    model_type = model_type or 'IndependentActorCriticMLP'
    model_kwargs = model_kwargs or {}
    MODEL_CLASSES = {
        'IndependentActorCriticMLP': IndependentActorCriticMLP
    }
    try:
        model_class = MODEL_CLASSES[model_type]
        return model_class(state_dim, action_dim, **model_kwargs)
    except KeyError:
        raise ValueError(f"Not supported model type: {model_type}")
from .mlp import C51MLP


def get_model(state_dim, action_dim, num_atoms, model_type=None, model_kwargs={}):
    model_type = model_type or 'C51MLP'
    model_kwargs = model_kwargs or {}
    MODEL_CLASSES = {
        'C51MLP': C51MLP
    }
    try:
        model_class = MODEL_CLASSES[model_type]
        return model_class(state_dim, action_dim, num_atoms, **model_kwargs)
    except KeyError:
        raise ValueError(f"Not supported model type: {model_type}")
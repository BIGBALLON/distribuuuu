try:
    from torch.hub import load_state_dict_from_url  # noqa
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa

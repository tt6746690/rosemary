from omegaconf import ListConfig, DictConfig, OmegaConf


__all__ = [
    "omegaconf_to_container",
]

def omegaconf_to_container(c):
    if isinstance(c, (ListConfig, DictConfig)):
        c = OmegaConf.to_container(c, resolve=True)
    return c
    
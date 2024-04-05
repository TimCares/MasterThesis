from hydra.utils import instantiate
from omegaconf import OmegaConf

def build_config(conf_path:str=None):
    cli_conf = OmegaConf.from_cli()
    if conf_path is None:
        if "config" not in cli_conf:
            raise ValueError(
                "Please pass 'config' to specify configuration yaml file"
            )
        yaml_conf = OmegaConf.load(cli_conf.config)
        cli_conf.pop("config")
    else:
        yaml_conf = OmegaConf.load(conf_path)
    conf = instantiate(yaml_conf)
    config = OmegaConf.merge(conf, cli_conf)
    return config
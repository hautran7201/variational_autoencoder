import textwrap

import hydra
from omegaconf import DictConfig

from train import train


# @hydra.main(version_base=None)
@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    """Hydra wrapper for train."""
    if not config:
        raise ValueError(
            textwrap.dedent("""\
                            Config path and name not specified!
                            Please specify these by using --config-path and --config-name, respectively."""))    

    return train(config.parameters)


if __name__ == '__main__':
    main()
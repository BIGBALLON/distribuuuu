import distribuuuu.config as config
import distribuuuu.trainer as trainer
from distribuuuu.config import cfg


def main():
    config.load_cfg_fom_args(description="Test a classification model.")
    cfg.freeze()
    trainer.test_model()


if __name__ == "__main__":
    main()

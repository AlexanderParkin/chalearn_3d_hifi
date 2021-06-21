from at_learner_core.trainer import Runner
from at_learner_core.datasets.dataset_manager import DatasetManager
from models.wrappers.fpwrapper import FPWrapper
class RGBRunner(Runner):
    def __init__(self, config, train=True):
        super().__init__(config, train=train)

    def _init_wrapper(self):
        if self.config.wrapper_config.wrapper_name == 'FPWrapper':
            self.wrapper = FPWrapper(self.config.wrapper_config)
        

            
        self.wrapper = self.wrapper.to(self.device)

    def _init_loaders(self):
        self.train_loader = DatasetManager.get_dataloader(self.config.datalist_config.trainlist_config,
                                                          self.config.train_process_config)

        self.val_loader = DatasetManager.get_dataloader(self.config.datalist_config.testlist_configs,
                                                        self.config.train_process_config,
                                                        shuffle=False)

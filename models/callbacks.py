from pytorch_lightning.callbacks import Callback

class TestResultCollector(Callback):
    def on_test_start(self, trainer, pl_module):
        trainer.test_outputs = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        trainer.test_outputs.append(outputs)
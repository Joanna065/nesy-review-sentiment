from pytorch_lightning import Callback


class ModelSize(Callback):
    def on_fit_start(self, trainer, pl_module):
        memory_params = sum(
            [param.nelement() * param.element_size() for param in pl_module.parameters()]
        )
        memory_buffers = sum([buf.nelement() * buf.element_size() for buf in pl_module.buffers()])
        memory_used = memory_params + memory_buffers  # in bytes

        parameters_num = 0
        for n, p in pl_module.named_parameters():
            parameters_num += p.nelement()

        trainer.logger.log_metrics({'num_parameters': parameters_num})
        trainer.logger.log_metrics({'memory_used': memory_used})

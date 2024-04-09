import os

from .dist import is_main_process

class TextLogger():
    def __init__(self, save, filename):
        if is_main_process():
            self.f = os.path.join(save, filename)

    def write(self, log, print_log=True):
        if is_main_process():
            if print_log:
                print(log)
            with open(self.f, 'a') as f:
                f.write(log+'\n')
            return f.close()


class TensorboardLogger():
    def __init__(self, log_dir):
        if is_main_process():
            from torch.utils.tensorboard import SummaryWriter
            self.tb_logger = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, name, value, current_iter):
        if is_main_process():
            self.tb_logger.add_scalar(name, value, current_iter)
        return

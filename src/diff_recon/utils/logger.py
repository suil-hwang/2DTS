import logging
import os
import sys
from pathlib import Path
from typing import TextIO, NamedTuple, Union, Optional
from multiprocessing import Queue, Process
import traceback
from enum import Enum
from torch.utils.tensorboard import SummaryWriter
import torch


class LogType(Enum):
    LOG = 1
    TB_SCALAR = 2
    TB_IMAGE = 3
    TB_HISTOGRAM = 4


class LogRecord(NamedTuple):
    type: LogType
    tag: str
    value: Union[float, torch.Tensor, logging.LogRecord]
    global_step: int
    bins: Union[str, int] = "auto"


class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def getLogger(name, log_file_dir: Optional[str] = None, stream: Optional[TextIO] = None, level=logging.DEBUG, isolated: bool = False) -> logging.Logger:
    logger = logging.Logger(name) if isolated else logging.getLogger(name)
    logger.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    stream_formatter = ColoredFormatter()

    if log_file_dir is not None:
        file_path = os.path.abspath(os.path.join(log_file_dir, f"{name}_outputs.log"))
        if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == file_path
                   for handler in logger.handlers):
            fileinfo = logging.FileHandler(file_path, encoding="utf-8")
            fileinfo.setLevel(logging.INFO)
            fileinfo.setFormatter(file_formatter)
            logger.addHandler(fileinfo)

    if stream is not None:
        if not any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
                   and handler.stream is stream for handler in logger.handlers):
            controlshow = logging.StreamHandler(stream)
            controlshow.setLevel(logging.DEBUG)
            controlshow.setFormatter(stream_formatter)
            logger.addHandler(controlshow)

    return logger


class Logger_MP:
    # handle logging in the background process
    def __init__(self, name, log_file_dir: Optional[str] = None, stream: Optional[TextIO] = sys.stdout, use_tensorboard: bool = True):
        if log_file_dir is not None:
            Path(log_file_dir).mkdir(parents=True, exist_ok=True)

        self.name = name
        self.use_tensorboard = use_tensorboard

        self.log_queue = Queue(-1)
        if stream is sys.stdout:
            stream = "__stdout__"
        elif stream is sys.stderr:
            stream = "__stderr__"
        self.listener = Process(target=Logger_MP._listener_process,
                                args=(name, self.log_queue, log_file_dir, stream, use_tensorboard))
        self.listener.start()

    @staticmethod
    def _listener_process(name, log_queue: Queue, log_file_dir: str, stream: TextIO, use_tensorboard: bool):
        if stream == "__stdout__":
            stream = sys.stdout
        elif stream == "__stderr__":
            stream = sys.stderr
        logger = getLogger(name, log_file_dir, stream, isolated=True)
        tb_writter = SummaryWriter(log_file_dir) if use_tensorboard and log_file_dir is not None else None

        while True:
            try:
                record = log_queue.get()
                if record is None:  # send None to tell the listener to exit
                    break

                if record.type == LogType.LOG:
                    logger.handle(record.value)
                elif record.type == LogType.TB_SCALAR:
                    if tb_writter is not None:
                        tb_writter.add_scalar(record.tag, record.value, record.global_step)
                elif record.type == LogType.TB_IMAGE:
                    if tb_writter is not None:
                        tb_writter.add_image(record.tag, record.value, record.global_step)
                elif record.type == LogType.TB_HISTOGRAM:
                    if tb_writter is not None:
                        tb_writter.add_histogram(record.tag, record.value, record.global_step, bins=record.bins)
            except Exception:
                logger.error("Whoops! Problem:")
                traceback.print_exc(file=sys.stderr)

        if tb_writter is not None:
            tb_writter.close()
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()

    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        if self.use_tensorboard:
            if isinstance(scalar_value, torch.Tensor):
                scalar_value = scalar_value.detach().cpu()
            log_record = LogRecord(LogType.TB_SCALAR, tag, scalar_value, global_step)
            self.log_queue.put(log_record)

    def add_image(self, tag: str, img_tensor, global_step: int):
        if self.use_tensorboard:
            if isinstance(img_tensor, torch.Tensor):
                img_tensor = img_tensor.detach().cpu()
            log_record = LogRecord(LogType.TB_IMAGE, tag, img_tensor, global_step)
            self.log_queue.put(log_record)

    def add_histogram(self, tag: str, values, global_step: int, bins="auto"):
        if self.use_tensorboard:
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu()
            log_record = LogRecord(LogType.TB_HISTOGRAM, tag, values, global_step, bins)
            self.log_queue.put(log_record)

    def log(self, level, msg, *args, **kwargs):
        record = logging.LogRecord(name=None, level=level, pathname=None, lineno=None, msg=msg, args=args, exc_info=None, func=None, sinfo=None)
        log_record = LogRecord(LogType.LOG, None, record, None)
        self.log_queue.put(log_record)

    def info(self, message: str):
        self.log(logging.INFO, message)

    def debug(self, message: str):
        self.log(logging.DEBUG, message)

    def warning(self, message: str):
        self.log(logging.WARNING, message)

    def error(self, message: str):
        self.log(logging.ERROR, message)

    def critical(self, message: str):
        self.log(logging.CRITICAL, message)

    def warnOnce(self, message: str):
        if not hasattr(self, "warned_msg"):
            self.warned_msg = set()
        if message not in self.warned_msg:
            self.warning(message)
            self.warned_msg.add(message)

    def __del__(self):
        listener = getattr(self, "listener", None)
        if listener is not None and listener.pid is not None:
            if listener.is_alive():
                self.log_queue.put(None)
            listener.join()


class Logger:
    # handle logging in the main process
    def __init__(self, name, log_file_dir: Optional[str] = None, stream: Optional[TextIO] = sys.stdout, use_tensorboard: bool = False):
        if log_file_dir is not None:
            Path(log_file_dir).mkdir(parents=True, exist_ok=True)

        self.name = name
        # Each wrapper owns its handlers; same-second experiments must not share sinks.
        self.logger = getLogger(name, log_file_dir, stream, isolated=True)
        self.tb_writer = SummaryWriter(log_file_dir) if use_tensorboard and log_file_dir is not None else None

    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, scalar_value, global_step)

    def add_image(self, tag: str, img_tensor, global_step: int):
        if self.tb_writer is not None:
            self.tb_writer.add_image(tag, img_tensor, global_step)

    def add_histogram(self, tag: str, values, global_step: int, bins="auto"):
        if self.tb_writer is not None:
            self.tb_writer.add_histogram(tag, values, global_step, bins)

    def log(self, level, msg, *args, **kwargs):
        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, message: str):
        self.log(logging.DEBUG, message)

    def info(self, message: str):
        self.log(logging.INFO, message)

    def warning(self, message: str):
        self.log(logging.WARNING, message)

    def error(self, message: str):
        self.log(logging.ERROR, message)

    def critical(self, message: str):
        self.log(logging.CRITICAL, message)

    def warnOnce(self, message: str):
        if not hasattr(self, "warned_msg"):
            self.warned_msg = set()
        if message not in self.warned_msg:
            self.warning(message)
            self.warned_msg.add(message)

    def __del__(self):
        writer = getattr(self, "tb_writer", None)
        if writer is not None:
            writer.close()
        logger = getattr(self, "logger", None)
        for handler in logger.handlers if logger is not None else ():
            try:
                handler.close()
            except Exception as e:
                pass


logging.lastResort = logging.NullHandler()
empty_logger = Logger("empty_logger", stream=None)
stdout_logger = Logger("stdout_logger")

import abc
from torch.utils.data import Dataset, DataLoader

from ..utils.config import Config
from ..utils.logger import Logger, stdout_logger
from ..utils.camera import Camera
from ..utils.file_handler import BaseFileHandler


def nop(x):
    return x


class BaseDatasetFactory(abc.ABC):
    def __init__(self, config: Config = None, logger: Logger = None):
        self._config = config if config is not None else Config()
        self._logger = logger if logger is not None else stdout_logger

        self._num_workers = config.num_workers if config.num_workers is not None else 1
        self._pin_memory = True

        self._train_dataset: Dataset = None
        self._test_dataset: Dataset = None
        self._file_handler: BaseFileHandler = None

    def getFileHandler(self) -> BaseFileHandler:
        if self._file_handler is None:
            raise ValueError("File handler is not initialized.")
        return self._file_handler

    def _getDataLoader(self, dataset: Dataset, shuffle: bool = True, num_workers: int = None):
        if len(dataset) == 0:
            return []

        num_workers = self._num_workers if num_workers is None else num_workers
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self._pin_memory,
            collate_fn=nop, 
            prefetch_factor=10 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,  
        )

    def getTrainDataset(self) -> DataLoader:
        if not hasattr(self, "_train_loader"):
            self._train_loader = self._getDataLoader(self._train_dataset)
        return iter(self._train_loader)

    def getTrainDatasetSize(self) -> int:
        return len(self._train_dataset)

    def getTrainData(self, idx) -> Camera:
        return self._train_dataset[idx]

    def nextTrainData(self) -> Camera:
        if not hasattr(self, "_train_dataloader"):
            self._train_dataloader = self.getTrainDataset()

        try:
            data = next(self._train_dataloader)
        except StopIteration:
            self._train_dataloader = self.getTrainDataset()
            data = next(self._train_dataloader)
        return data

    def getTestDataset(self) -> DataLoader:
        if not hasattr(self, "_test_loader"):
            self._test_loader = self._getDataLoader(self._test_dataset, shuffle=False)
        return iter(self._test_loader)

    def getTestDatasetSize(self) -> int:
        return len(self._test_dataset)

    def getTestData(self, idx) -> Camera:
        return self._test_dataset[idx]

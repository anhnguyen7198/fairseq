
from typing import Callable
from . import BaseWrapperDataset


class AssertDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.assertions = []

    def add_assertion(self, assertion: Callable, message: str):
        self.assertions.append((assertion, message))

    def __getitem__(self, index):
        item = self.dataset[index]
        for assertion in self.assertions:
            assert assertion[0](item), assertion[1]
        return item

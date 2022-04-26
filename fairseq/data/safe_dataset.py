from . import BaseWrapperDataset, FairseqDataset


class SafeDataset(BaseWrapperDataset):
    def __init__(self, dataset: FairseqDataset, safe_collate: bool = True, rounds_index_selection: int = 0):
        super().__init__(dataset)
        self.safe_collate = safe_collate
        self.rounds_index_selection = rounds_index_selection

    def __getitem__(self, index):
        try:
            return self.dataset[index]
        except BaseException:
            for i in range(self.rounds_index_selection):
                try:
                    return self.dataset[(index + i) % len(self)]
                except BaseException:
                    continue
            return None

    def collater(self, samples):
        samples = [x for x in samples if x is not None]
        return super().collater(samples)

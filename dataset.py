import numpy as np
import torch
import yfinance as yf
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """Stock time series dataset, downloads from Yahoo Finance."""

    def __init__(
        self,
        ticker: str = "D05.SI",
        start: str = "2020-01-01",
        end=None,
        normalize: bool = True,
    ) -> None:
        """
        Args:
            ticker (string): Yahoo Finance ticker symbol (e.g., 'D05.SI' for DBS Group)
            start (string): start date in 'YYYY-MM-DD' format (optional)
            end (string): end date in 'YYYY-MM-DD' format (optional)
            normalize (bool): whether to normalize the data in [-1,1]
        """
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, multi_level_index=False)
        if df is None or df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        df = df[["Close"]].copy()
        df["Timestamp"] = df.index
        df = df.set_index("Timestamp")
        close_price = df["Close"]
        data = torch.from_numpy(
            np.expand_dims(np.array([group[1] for group in close_price.groupby(df.index.date)]), -1)
        ).float()
        self.data = self.normalize(data) if normalize else data
        self.seq_len = data.size(1)

        # Estimates distribution parameters of deltas (Gaussian) from normalized data
        original_deltas = data[:, -1] - data[:, 0]
        self.original_deltas = original_deltas
        self.or_delta_max, self.or_delta_min = original_deltas.max(), original_deltas.min()
        deltas = self.data[:, -1] - self.data[:, 0]
        self.deltas = deltas
        self.delta_mean, self.delta_std = deltas.mean(), deltas.std()
        self.delta_max, self.delta_min = deltas.max(), deltas.min()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    def normalize(self, x) -> torch.Tensor:
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        self.max = x.max()
        self.min = x.min()
        return 2 * (x - self.min) / (self.max - self.min) - 1

    def denormalize(self, x):
        """Revert [-1,1] normalization"""
        if not hasattr(self, "max") or not hasattr(self, "min"):
            raise Exception("You are calling denormalize, but the input was not normalized")
        return 0.5 * (x * self.max - x * self.min + self.max + self.min)

    def sample_deltas(self, number) -> torch.Tensor:
        """Sample a vector of (number) deltas from the fitted Gaussian"""
        return (torch.randn(number, 1) + self.delta_mean) * self.delta_std

    def normalize_deltas(self, x):
        return (self.delta_max - self.delta_min) * (x - self.or_delta_min) / (
            self.or_delta_max - self.or_delta_min
        ) + self.delta_min

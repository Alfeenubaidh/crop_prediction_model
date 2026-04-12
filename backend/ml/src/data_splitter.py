# src/data_splitter.py

import pandas as pd
import yaml


class DataSplitterConfig:
    def __init__(self, config_path="config.yaml", cfg: dict | None = None):
        if cfg is None:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

        split_cfg = cfg.get("data_split", {})

        if "train_end_year" in split_cfg:
            self.train_end_year = split_cfg["train_end_year"]
        elif "train_years" in split_cfg:
            self.train_end_year = max(split_cfg["train_years"])
        else:
            self.train_end_year = 2018

        if "val_end_year" in split_cfg:
            self.val_end_year = split_cfg["val_end_year"]
        elif "val_years" in split_cfg:
            self.val_end_year = max(split_cfg["val_years"])
        else:
            self.val_end_year = 2020

        self.target_column = cfg["data_sources"]["yield"].get(
            "target_column", "Yield"
        )

        self.year_column = split_cfg.get("year_column", "Year")


class DataSplitter:
    def __init__(self, config: DataSplitterConfig):
        self.config = config

    def split(self, df: pd.DataFrame) -> dict:
        df = df.copy()

        target = self.config.target_column
        year_col = self.config.year_column

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' NOT found in dataframe!")

        if year_col not in df.columns:
            raise ValueError(f"Year column '{year_col}' NOT found in dataframe!")

        # Ensure sorted by time
        df = df.sort_values(year_col)

        # Split by year
        train_df = df[df[year_col] <= self.config.train_end_year]
        val_df = df[
            (df[year_col] > self.config.train_end_year)
            & (df[year_col] <= self.config.val_end_year)
        ]
        test_df = df[df[year_col] > self.config.val_end_year]

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(
                "One of train/val/test splits is EMPTY. "
                "Check year boundaries and data coverage."
            )

        return {
            "X_train": train_df.drop(columns=[target]),
            "y_train": train_df[target],
            "X_val": val_df.drop(columns=[target]),
            "y_val": val_df[target],
            "X_test": test_df.drop(columns=[target]),
            "y_test": test_df[target],
        }

    def split_full(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Same year boundaries as split(), but returns full rows including the target column.
        Use this when downstream steps (e.g. lagged Yield) need the label on training history.
        """
        df = df.copy()
        target = self.config.target_column
        year_col = self.config.year_column

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' NOT found in dataframe!")
        if year_col not in df.columns:
            raise ValueError(f"Year column '{year_col}' NOT found in dataframe!")

        df = df.sort_values(year_col)

        train_df = df[df[year_col] <= self.config.train_end_year]
        val_df = df[
            (df[year_col] > self.config.train_end_year)
            & (df[year_col] <= self.config.val_end_year)
        ]
        test_df = df[df[year_col] > self.config.val_end_year]

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(
                "One of train/val/test splits is EMPTY. "
                "Check year boundaries and data coverage."
            )

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(
            drop=True
        )

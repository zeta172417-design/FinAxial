from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_start: int
    train_end: int
    validation_dates: tuple[int, ...]
    evaluation_year: int
    evaluation_dates: tuple[int, ...]
    excluded_boundary_date: int
    new_year: int | None

    @property
    def fit_end(self) -> int:
        return self.validation_dates[0] - 1


@dataclass(frozen=True)
class Phase1Split:
    tuning_train_dates: tuple[int, ...]
    validation_dates: tuple[int, ...]
    final_train_dates: tuple[int, ...]
    test_dates: tuple[int, ...]
    boundary_excluded_date: int


def make_phase1_split(
    dates: np.ndarray,
    *,
    tuning_train_end: int = 20241104,
    boundary_excluded_date: int = 20241105,
    validation_start: int = 20241106,
    validation_end: int = 20241231,
    final_train_end: int = 20241231,
    test_start: int = 20250102,
    test_end: int = 20260605,
    expected_validation_days: int = 40,
) -> Phase1Split:
    dates = np.asarray(dates, dtype=np.int64)
    if dates.ndim != 1 or not len(dates) or not np.all(dates[:-1] < dates[1:]):
        raise ValueError("dates must be a non-empty, strictly increasing 1-D array")
    date_set = set(int(value) for value in dates)
    if boundary_excluded_date not in date_set:
        raise ValueError(f"boundary date {boundary_excluded_date} is missing")
    tuning = dates[dates <= tuning_train_end]
    validation = dates[(dates >= validation_start) & (dates <= validation_end)]
    final_train = dates[dates <= final_train_end]
    test = dates[(dates >= test_start) & (dates <= test_end)]
    if len(validation) != expected_validation_days:
        raise ValueError(f"expected {expected_validation_days} validation dates, found {len(validation)}")
    if not len(tuning) or not len(final_train) or not len(test):
        raise ValueError("one or more phase-1 split segments are empty")
    split = Phase1Split(
        tuning_train_dates=tuple(int(x) for x in tuning),
        validation_dates=tuple(int(x) for x in validation),
        final_train_dates=tuple(int(x) for x in final_train),
        test_dates=tuple(int(x) for x in test),
        boundary_excluded_date=int(boundary_excluded_date),
    )
    assert_phase1_disjoint(split)
    return split


def assert_phase1_disjoint(split: Phase1Split) -> None:
    tuning = set(split.tuning_train_dates)
    validation = set(split.validation_dates)
    test = set(split.test_dates)
    if tuning & validation or tuning & test or validation & test:
        raise AssertionError("phase-1 train/validation/test leakage")
    if split.boundary_excluded_date in tuning | validation | test:
        raise AssertionError("label-leaking boundary date was not isolated")
    if not tuning.issubset(set(split.final_train_dates)) or not validation.issubset(set(split.final_train_dates)):
        raise AssertionError("final retraining must contain tuning train and validation dates")


def make_folds(dates: np.ndarray, validation_days: int = 40) -> list[WalkForwardFold]:
    dates = np.asarray(dates, dtype=np.int64)
    if dates.ndim != 1 or not np.all(dates[:-1] < dates[1:]):
        raise ValueError("dates must be a strictly increasing 1-D array")
    folds: list[WalkForwardFold] = []
    for evaluation_year in (2021, 2022, 2023, 2024):
        eval_dates = dates[dates // 10000 == evaluation_year]
        earlier = dates[dates < evaluation_year * 10000]
        if not len(eval_dates) or len(earlier) <= validation_days:
            raise ValueError(f"insufficient dates for {evaluation_year}")
        # y_ret_1d on the last pre-evaluation trading date uses the first evaluation
        # close, so that boundary sample is excluded from both fit and validation.
        boundary = int(earlier[-1])
        usable = earlier[:-1]
        validation = usable[-validation_days:]
        fit = usable[:-validation_days]
        folds.append(WalkForwardFold(
            name="calibration" if evaluation_year == 2021 else f"fold-{evaluation_year - 2021}",
            train_start=int(fit[0]),
            train_end=int(fit[-1]),
            validation_dates=tuple(int(x) for x in validation),
            evaluation_year=evaluation_year,
            evaluation_dates=tuple(int(x) for x in eval_dates),
            excluded_boundary_date=boundary,
            new_year=None if evaluation_year == 2021 else evaluation_year - 1,
        ))
    return folds


def assert_disjoint(fold: WalkForwardFold) -> None:
    train = set(range(fold.train_start, fold.train_end + 1))
    valid = set(fold.validation_dates)
    evaluation = set(fold.evaluation_dates)
    if train & valid or train & evaluation or valid & evaluation:
        raise AssertionError(f"date leakage in {fold.name}")
    if fold.excluded_boundary_date in valid or fold.excluded_boundary_date in evaluation:
        raise AssertionError("boundary date was not isolated")

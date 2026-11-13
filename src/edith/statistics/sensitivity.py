"""Which inputs drive a campaign's outputs, and its failures.

Spearman's rank correlation of each drawn input with each output measures how
monotonically the output rises or falls with that input alone; it does not
see interactions between inputs. Columns are those of
``edith.statistics.results.load_table``.
"""

import warnings
from typing import NamedTuple

import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]

__all__ = ["Sensitivity", "failure_drivers", "spearman"]


class Sensitivity(NamedTuple):
    """Rank correlations, one row per input and one column per output."""

    correlation: pd.DataFrame  # From -1 to 1; NaN for an input that never varies
    p_value: pd.DataFrame  # Of no correlation


def spearman(inputs: pd.DataFrame, outputs: pd.DataFrame) -> Sensitivity:
    """Return the rank correlation of every input with every output.

    Each output leaves out the trials where it, or an input, is NaN, such as
    failed trials.

    Args:
        inputs (pd.DataFrame): One column per drawn input, one row per trial.
        outputs (pd.DataFrame): One column per output, of the same trials.

    Returns:
        Sensitivity: The correlations and their p-values.
    """
    correlation = pd.DataFrame(
        index=inputs.columns, columns=outputs.columns, dtype=float
    )
    p_value = correlation.copy()
    for output in outputs.columns:
        valid = outputs[output].notna() & inputs.notna().all(axis=1)
        for key in inputs.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # An input that never varies
                result = stats.spearmanr(
                    inputs.loc[valid, key], outputs.loc[valid, output]
                )
            correlation.loc[key, output] = float(result.statistic)
            p_value.loc[key, output] = float(result.pvalue)
    return Sensitivity(correlation, p_value)


def failure_drivers(table: pd.DataFrame, inputs: list[str]) -> pd.DataFrame:
    """Return the inputs ranked by how strongly they go with a trial failing.

    Args:
        table (pd.DataFrame): The campaign's table, with its ``failed`` column.
        inputs (list[str]): Its input columns, as ``results.input_keys`` gives.

    Returns:
        pd.DataFrame: ``correlation`` and ``p_value`` of each input with
            failing, strongest first; NaN if no trial or every trial failed.
    """
    failed = table[["failed"]].astype(float)
    result = spearman(table[inputs], failed)
    ranked = pd.DataFrame(
        {
            "correlation": result.correlation["failed"],
            "p_value": result.p_value["failed"],
        }
    )
    return ranked.loc[ranked["correlation"].abs().sort_values(ascending=False).index]

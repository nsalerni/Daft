from __future__ import annotations

import pytest

from daft import col
from daft.recordbatch.micropartition import MicroPartition


class TestFillNullStrategy:
    @pytest.mark.parametrize(
        "strategy,input_data,expected",
        [
            # Forward strategy tests (pandas ffill behavior)
            ("forward", [None, 1, None], [None, 1, 1]),
            ("forward", [1, None, None], [1, 1, 1]),
            ("forward", [None, None, None], [None, None, None]),
            ("forward", [1, 2, 3], [1, 2, 3]),
            ("forward", [None, 1, None, 2, None], [None, 1, 1, 2, 2]),
            # Backward strategy tests (pandas bfill behavior)
            ("backward", [None, 1, None], [1, 1, None]),
            ("backward", [1, None, None], [1, None, None]),
            ("backward", [None, None, None], [None, None, None]),
            ("backward", [1, 2, 3], [1, 2, 3]),
            ("backward", [None, 1, None, 2, None], [1, 1, 2, 2, None]),
        ],
    )
    def test_fill_null_strategy_int(self, strategy, input_data, expected):
        mp = MicroPartition.from_pydict({"col": input_data})
        out = mp.eval_expression_list([col("col").fill_null(strategy=strategy)])
        assert out.to_pydict()["col"] == expected

    @pytest.mark.parametrize(
        "strategy,input_data,expected",
        [
            # Forward strategy tests (pandas ffill behavior)
            ("forward", [None, "a", None], [None, "a", "a"]),
            ("forward", ["a", None, None], ["a", "a", "a"]),
            ("forward", [None, None, None], [None, None, None]),
            ("forward", ["a", "b", "c"], ["a", "b", "c"]),
            # Backward strategy tests (pandas bfill behavior)
            ("backward", [None, "a", None], ["a", "a", None]),
            ("backward", ["a", None, None], ["a", None, None]),
            ("backward", [None, None, None], [None, None, None]),
            ("backward", ["a", "b", "c"], ["a", "b", "c"]),
        ],
    )
    def test_fill_null_strategy_string(self, strategy, input_data, expected):
        mp = MicroPartition.from_pydict({"col": input_data})
        out = mp.eval_expression_list([col("col").fill_null(strategy=strategy)])
        assert out.to_pydict()["col"] == expected

    @pytest.mark.parametrize(
        "strategy,input_data,expected",
        [
            # Forward strategy tests (pandas ffill behavior)
            ("forward", [None, True, None], [None, True, True]),
            ("forward", [False, None, None], [False, False, False]),
            ("forward", [None, None, None], [None, None, None]),
            ("forward", [True, False, True], [True, False, True]),
            # Backward strategy tests (pandas bfill behavior)
            ("backward", [None, True, None], [True, True, None]),
            ("backward", [False, None, None], [False, None, None]),
            ("backward", [None, None, None], [None, None, None]),
            ("backward", [True, False, True], [True, False, True]),
        ],
    )
    def test_fill_null_strategy_bool(self, strategy, input_data, expected):
        mp = MicroPartition.from_pydict({"col": input_data})
        out = mp.eval_expression_list([col("col").fill_null(strategy=strategy)])
        assert out.to_pydict()["col"] == expected

    @pytest.mark.parametrize("strategy", ["forward", "backward"])
    def test_fill_null_strategy_empty_array(self, strategy):
        mp = MicroPartition.from_pydict({"col": []})
        out = mp.eval_expression_list([col("col").fill_null(strategy=strategy)])
        assert out.to_pydict()["col"] == []

    @pytest.mark.parametrize("strategy", ["forward", "backward"])
    def test_fill_null_strategy_single_element(self, strategy):
        # Single non-null element
        mp = MicroPartition.from_pydict({"col": [1]})
        out = mp.eval_expression_list([col("col").fill_null(strategy=strategy)])
        assert out.to_pydict()["col"] == [1]

        # Single null element
        mp = MicroPartition.from_pydict({"col": [None]})
        out = mp.eval_expression_list([col("col").fill_null(strategy=strategy)])
        assert out.to_pydict()["col"] == [None]

    def test_fill_null_with_value_still_works(self):
        mp = MicroPartition.from_pydict({"col": [None, 1, None]})
        out = mp.eval_expression_list([col("col").fill_null(999)])
        assert out.to_pydict()["col"] == [999, 1, 999]

    def test_fill_null_strategy_preserves_dtype(self):
        mp = MicroPartition.from_pydict({"col": [None, 1.5, None]})
        out = mp.eval_expression_list([col("col").fill_null(strategy="forward")])
        result = out.to_pydict()["col"]
        assert result == [None, 1.5, 1.5]  # Pandas ffill behavior
        # Ensure it's still float (check the non-null values)
        assert result[0] is None
        assert isinstance(result[1], float)
        assert isinstance(result[2], float)

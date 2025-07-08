from __future__ import annotations

from daft import col
from daft.recordbatch.micropartition import MicroPartition
from daft.series import Series


class TestSeriesFillStrategy:
    """Test fill_null strategies at the Series level using expression roundtrips."""

    def test_fill_null_strategy_forward_series(self):
        s = Series.from_pylist([None, 2, None, 4, None])
        mp = MicroPartition.from_pydict({"x": s})
        filled = mp.eval_expression_list([col("x").fill_null(strategy="forward")]).to_pydict()["x"]
        assert filled == [None, 2, 2, 4, 4]  # Actual implementation behavior

    def test_fill_null_strategy_backward_series(self):
        s = Series.from_pylist([None, 2, None, 4, None])
        mp = MicroPartition.from_pydict({"x": s})
        filled = mp.eval_expression_list([col("x").fill_null(strategy="backward")]).to_pydict()["x"]
        assert filled == [2, 2, 4, 4, None]  # Actual implementation behavior

    def test_fill_null_strategy_string_series(self):
        s = Series.from_pylist([None, "hello", None, "world", None])
        mp = MicroPartition.from_pydict({"x": s})

        # Forward fill
        filled_forward = mp.eval_expression_list([col("x").fill_null(strategy="forward")]).to_pydict()["x"]
        assert filled_forward == [None, "hello", "hello", "world", "world"]  # Actual implementation behavior

        # Backward fill
        filled_backward = mp.eval_expression_list([col("x").fill_null(strategy="backward")]).to_pydict()["x"]
        assert filled_backward == ["hello", "hello", "world", "world", None]  # Actual implementation behavior

    def test_fill_null_strategy_all_null_series(self):
        s = Series.from_pylist([None, None, None])
        mp = MicroPartition.from_pydict({"x": s})

        # Both strategies should preserve all nulls
        filled_forward = mp.eval_expression_list([col("x").fill_null(strategy="forward")]).to_pydict()["x"]
        assert filled_forward == [None, None, None]

        filled_backward = mp.eval_expression_list([col("x").fill_null(strategy="backward")]).to_pydict()["x"]
        assert filled_backward == [None, None, None]

    def test_fill_null_strategy_no_nulls_series(self):
        s = Series.from_pylist([1, 2, 3])
        mp = MicroPartition.from_pydict({"x": s})

        # Both strategies should preserve the original data
        filled_forward = mp.eval_expression_list([col("x").fill_null(strategy="forward")]).to_pydict()["x"]
        assert filled_forward == [1, 2, 3]

        filled_backward = mp.eval_expression_list([col("x").fill_null(strategy="backward")]).to_pydict()["x"]
        assert filled_backward == [1, 2, 3]

    def test_fill_null_strategy_mixed_types_numeric(self):
        s = Series.from_pylist([None, 1.5, None, 2.7, None])
        mp = MicroPartition.from_pydict({"x": s})

        filled_forward = mp.eval_expression_list([col("x").fill_null(strategy="forward")]).to_pydict()["x"]
        assert filled_forward == [None, 1.5, 1.5, 2.7, 2.7]  # Actual implementation behavior

        filled_backward = mp.eval_expression_list([col("x").fill_null(strategy="backward")]).to_pydict()["x"]
        assert filled_backward == [1.5, 1.5, 2.7, 2.7, None]  # Actual implementation behavior

from __future__ import annotations

import pyarrow as pa
import pytest

from daft.series import Series


@pytest.mark.parametrize(
    "input,fill_value,expected",
    [
        # No broadcast
        [[1, 2, None], [3, 3, 3], [1, 2, 3]],
        # Broadcast input
        [[None], [3, 3, 3], [3, 3, 3]],
        # Broadcast fill_value
        [[1, 2, None], [3], [1, 2, 3]],
        # Empty
        [[], [], []],
    ],
)
def test_series_fill_null(input, fill_value, expected) -> None:
    s = Series.from_arrow(pa.array(input, pa.int64()))
    fill_value = Series.from_arrow(pa.array(fill_value, pa.int64()))
    filled = s.fill_null(fill_value)
    assert filled.to_pylist() == expected


def test_series_fill_null_bad_input() -> None:
    s = Series.from_arrow(pa.array([1, 2, 3], pa.int64()))
    with pytest.raises(ValueError, match="expected another Series but got"):
        s.fill_null([1, 2, 3])


@pytest.mark.parametrize(
    "strategy,input_data,expected",
    [
        # Forward strategy tests (actual implementation behavior)
        ("forward", [None, 1, None], [None, 1, 1]),
        ("forward", [1, None, None], [1, 1, 1]),
        ("forward", [None, None, None], [None, None, None]),
        ("forward", [1, 2, 3], [1, 2, 3]),
        ("forward", [None, 1, None, 2, None], [None, 1, 1, 2, 2]),
        ("forward", [], []),
        ("forward", [None], [None]),
        ("forward", [42], [42]),
        # Backward strategy tests (actual implementation behavior)
        ("backward", [None, 1, None], [1, 1, None]),
        ("backward", [1, None, None], [1, None, None]),
        ("backward", [None, None, None], [None, None, None]),
        ("backward", [1, 2, 3], [1, 2, 3]),
        ("backward", [None, 1, None, 2, None], [1, 1, 2, 2, None]),
        ("backward", [], []),
        ("backward", [None], [None]),
        ("backward", [42], [42]),
    ],
)
def test_series_fill_null_strategy_int(strategy, input_data, expected) -> None:
    """Test Series.fill_null with strategy parameter using int data."""
    from daft import col
    from daft.recordbatch.micropartition import MicroPartition

    # Use expression evaluation since Series doesn't expose strategy directly
    mp = MicroPartition.from_pydict({"col": input_data})
    result_mp = mp.eval_expression_list([col("col").fill_null(strategy=strategy)])
    result = result_mp.to_pydict()["col"]
    assert result == expected


@pytest.mark.parametrize(
    "strategy,input_data,expected",
    [
        # Forward strategy tests (actual implementation behavior)
        ("forward", [None, "a", None], [None, "a", "a"]),
        ("forward", ["a", None, None], ["a", "a", "a"]),
        ("forward", [None, None, None], [None, None, None]),
        ("forward", ["a", "b", "c"], ["a", "b", "c"]),
        # Backward strategy tests (actual implementation behavior)
        ("backward", [None, "a", None], ["a", "a", None]),
        ("backward", ["a", None, None], ["a", None, None]),
        ("backward", [None, None, None], [None, None, None]),
        ("backward", ["a", "b", "c"], ["a", "b", "c"]),
    ],
)
def test_series_fill_null_strategy_string(strategy, input_data, expected) -> None:
    """Test Series.fill_null with strategy parameter using string data."""
    from daft import col
    from daft.recordbatch.micropartition import MicroPartition

    # Use expression evaluation since Series doesn't expose strategy directly
    mp = MicroPartition.from_pydict({"col": input_data})
    result_mp = mp.eval_expression_list([col("col").fill_null(strategy=strategy)])
    result = result_mp.to_pydict()["col"]
    assert result == expected


@pytest.mark.parametrize(
    "strategy,input_data,expected",
    [
        # Forward strategy tests (actual implementation behavior)
        ("forward", [None, True, None], [None, True, True]),
        ("forward", [False, None, None], [False, False, False]),
        ("forward", [None, None, None], [None, None, None]),
        ("forward", [True, False, True], [True, False, True]),
        # Backward strategy tests (actual implementation behavior)
        ("backward", [None, True, None], [True, True, None]),
        ("backward", [False, None, None], [False, None, None]),
        ("backward", [None, None, None], [None, None, None]),
        ("backward", [True, False, True], [True, False, True]),
    ],
)
def test_series_fill_null_strategy_bool(strategy, input_data, expected) -> None:
    """Test Series.fill_null with strategy parameter using boolean data."""
    from daft import col
    from daft.recordbatch.micropartition import MicroPartition

    # Use expression evaluation since Series doesn't expose strategy directly
    mp = MicroPartition.from_pydict({"col": input_data})
    result_mp = mp.eval_expression_list([col("col").fill_null(strategy=strategy)])
    result = result_mp.to_pydict()["col"]
    assert result == expected


def test_series_fill_null_strategy_preserves_dtype() -> None:
    """Test that strategy-based fill_null preserves the original dtype."""
    from daft import col
    from daft.recordbatch.micropartition import MicroPartition

    # Test with float data to ensure dtype preservation
    input_data = [None, 1.5, None, 2.7, None]
    mp = MicroPartition.from_pydict({"col": input_data})

    result_forward = mp.eval_expression_list([col("col").fill_null(strategy="forward")]).to_pydict()["col"]
    result_backward = mp.eval_expression_list([col("col").fill_null(strategy="backward")]).to_pydict()["col"]

    assert result_forward == [None, 1.5, 1.5, 2.7, 2.7]  # Actual implementation behavior
    assert result_backward == [1.5, 1.5, 2.7, 2.7, None]  # Actual implementation behavior

    # Ensure values are still floats
    assert isinstance(result_forward[1], float)
    assert isinstance(result_backward[1], float)

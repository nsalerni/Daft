from __future__ import annotations

import datetime

import pytest

from daft.datatype import DataType
from daft.expressions.expressions import col
from daft.recordbatch.micropartition import MicroPartition


@pytest.mark.parametrize(
    "input,fill_value,expected",
    [
        pytest.param([None, None, None], "a", ["a", "a", "a"], id="NullColumn"),
        pytest.param([True, False, None], False, [True, False, False], id="BoolColumn"),
        pytest.param(["a", "b", None], "b", ["a", "b", "b"], id="StringColumn"),
        pytest.param([b"a", None, b"c"], b"b", [b"a", b"b", b"c"], id="BinaryColumn"),
        pytest.param([-1, None, 3], 0, [-1, 0, 3], id="IntColumn"),
        pytest.param([-1.0, None, 3.0], 0.0, [-1.0, 0.0, 3.0], id="FloatColumn"),
        pytest.param(
            [datetime.date.today(), None, datetime.date(2023, 1, 1)],
            datetime.date(2022, 1, 1),
            [
                datetime.date.today(),
                datetime.date(2022, 1, 1),
                datetime.date(2023, 1, 1),
            ],
        ),
        pytest.param(
            [datetime.datetime(2022, 1, 1), None, datetime.datetime(2023, 1, 1)],
            datetime.datetime(2022, 1, 1),
            [
                datetime.datetime(2022, 1, 1),
                datetime.datetime(2022, 1, 1),
                datetime.datetime(2023, 1, 1),
            ],
        ),
    ],
)
def test_table_expr_fill_null(input, fill_value, expected) -> None:
    daft_recordbatch = MicroPartition.from_pydict({"input": input})
    daft_recordbatch = daft_recordbatch.eval_expression_list([col("input").fill_null(fill_value)])
    pydict = daft_recordbatch.to_pydict()

    assert pydict["input"] == expected


@pytest.mark.parametrize(
    "float_dtype",
    [DataType.float32(), DataType.float64()],
)
def test_table_expr_fill_nan(float_dtype) -> None:
    input = [1.0, None, 3.0, float("nan")]
    fill_value = 2.0
    expected = [1.0, None, 3.0, 2.0]

    daft_recordbatch = MicroPartition.from_pydict({"input": input})
    daft_recordbatch = daft_recordbatch.eval_expression_list(
        [col("input").cast(float_dtype).float.fill_nan(fill_value)]
    )
    pydict = daft_recordbatch.to_pydict()

    assert pydict["input"] == expected


@pytest.mark.parametrize(
    "strategy,input_data,expected",
    [
        # Forward strategy tests (actual implementation behavior)
        pytest.param("forward", [None, 1, None], [None, 1, 1], id="forward_int_mixed"),
        pytest.param("forward", [1, None, None], [1, 1, 1], id="forward_int_trailing_nulls"),
        pytest.param("forward", [None, None, None], [None, None, None], id="forward_int_all_nulls"),
        pytest.param("forward", [1, 2, 3], [1, 2, 3], id="forward_int_no_nulls"),
        pytest.param("forward", [], [], id="forward_int_empty"),
        pytest.param("forward", [None], [None], id="forward_int_single_null"),
        pytest.param("forward", [42], [42], id="forward_int_single_value"),
        # Backward strategy tests (actual implementation behavior)
        pytest.param("backward", [None, 1, None], [1, 1, None], id="backward_int_mixed"),
        pytest.param("backward", [1, None, None], [1, None, None], id="backward_int_trailing_nulls"),
        pytest.param("backward", [None, None, None], [None, None, None], id="backward_int_all_nulls"),
        pytest.param("backward", [1, 2, 3], [1, 2, 3], id="backward_int_no_nulls"),
        pytest.param("backward", [], [], id="backward_int_empty"),
        pytest.param("backward", [None], [None], id="backward_int_single_null"),
        pytest.param("backward", [42], [42], id="backward_int_single_value"),
    ],
)
def test_table_expr_fill_null_strategy_int(strategy, input_data, expected) -> None:
    """Test fill_null with forward/backward strategies for integer data."""
    daft_recordbatch = MicroPartition.from_pydict({"input": input_data})
    daft_recordbatch = daft_recordbatch.eval_expression_list([col("input").fill_null(strategy=strategy)])
    pydict = daft_recordbatch.to_pydict()
    assert pydict["input"] == expected


@pytest.mark.parametrize(
    "strategy,input_data,expected",
    [
        # Forward strategy tests (actual implementation behavior)
        pytest.param("forward", [None, "a", None], [None, "a", "a"], id="forward_str_mixed"),
        pytest.param("forward", ["a", None, None], ["a", "a", "a"], id="forward_str_trailing_nulls"),
        pytest.param("forward", [None, None, None], [None, None, None], id="forward_str_all_nulls"),
        pytest.param("forward", ["a", "b", "c"], ["a", "b", "c"], id="forward_str_no_nulls"),
        # Backward strategy tests (actual implementation behavior)
        pytest.param("backward", [None, "a", None], ["a", "a", None], id="backward_str_mixed"),
        pytest.param("backward", ["a", None, None], ["a", None, None], id="backward_str_trailing_nulls"),
        pytest.param("backward", [None, None, None], [None, None, None], id="backward_str_all_nulls"),
        pytest.param("backward", ["a", "b", "c"], ["a", "b", "c"], id="backward_str_no_nulls"),
    ],
)
def test_table_expr_fill_null_strategy_string(strategy, input_data, expected) -> None:
    """Test fill_null with forward/backward strategies for string data."""
    daft_recordbatch = MicroPartition.from_pydict({"input": input_data})
    daft_recordbatch = daft_recordbatch.eval_expression_list([col("input").fill_null(strategy=strategy)])
    pydict = daft_recordbatch.to_pydict()
    assert pydict["input"] == expected


@pytest.mark.parametrize(
    "strategy,input_data,expected",
    [
        # Forward strategy tests (actual implementation behavior)
        pytest.param("forward", [None, True, None], [None, True, True], id="forward_bool_mixed"),
        pytest.param("forward", [False, None, None], [False, False, False], id="forward_bool_trailing_nulls"),
        pytest.param("forward", [None, None, None], [None, None, None], id="forward_bool_all_nulls"),
        pytest.param("forward", [True, False, True], [True, False, True], id="forward_bool_no_nulls"),
        # Backward strategy tests (actual implementation behavior)
        pytest.param("backward", [None, True, None], [True, True, None], id="backward_bool_mixed"),
        pytest.param("backward", [False, None, None], [False, None, None], id="backward_bool_trailing_nulls"),
        pytest.param("backward", [None, None, None], [None, None, None], id="backward_bool_all_nulls"),
        pytest.param("backward", [True, False, True], [True, False, True], id="backward_bool_no_nulls"),
    ],
)
def test_table_expr_fill_null_strategy_bool(strategy, input_data, expected) -> None:
    """Test fill_null with forward/backward strategies for boolean data."""
    daft_recordbatch = MicroPartition.from_pydict({"input": input_data})
    daft_recordbatch = daft_recordbatch.eval_expression_list([col("input").fill_null(strategy=strategy)])
    pydict = daft_recordbatch.to_pydict()
    assert pydict["input"] == expected


@pytest.mark.parametrize(
    "strategy,input_data,expected",
    [
        # Forward strategy tests (actual implementation behavior)
        pytest.param("forward", [None, 1.5, None], [None, 1.5, 1.5], id="forward_float_mixed"),
        pytest.param("forward", [1.5, None, None], [1.5, 1.5, 1.5], id="forward_float_trailing_nulls"),
        # Backward strategy tests (actual implementation behavior)
        pytest.param("backward", [None, 1.5, None], [1.5, 1.5, None], id="backward_float_mixed"),
        pytest.param("backward", [1.5, None, None], [1.5, None, None], id="backward_float_trailing_nulls"),
    ],
)
def test_table_expr_fill_null_strategy_float(strategy, input_data, expected) -> None:
    """Test fill_null with forward/backward strategies for float data."""
    daft_recordbatch = MicroPartition.from_pydict({"input": input_data})
    daft_recordbatch = daft_recordbatch.eval_expression_list([col("input").fill_null(strategy=strategy)])
    pydict = daft_recordbatch.to_pydict()
    assert pydict["input"] == expected


def test_table_expr_fill_null_strategy_vs_value() -> None:
    """Test that value-based fill_null still works alongside strategy parameter."""
    input_data = [None, 1, None]

    # Test value-based fill
    daft_recordbatch = MicroPartition.from_pydict({"input": input_data})
    result_value = daft_recordbatch.eval_expression_list([col("input").fill_null(999)])
    assert result_value.to_pydict()["input"] == [999, 1, 999]

    # Test strategy-based fill
    result_forward = daft_recordbatch.eval_expression_list([col("input").fill_null(strategy="forward")])
    assert result_forward.to_pydict()["input"] == [None, 1, 1]  # Actual implementation behavior

    result_backward = daft_recordbatch.eval_expression_list([col("input").fill_null(strategy="backward")])
    assert result_backward.to_pydict()["input"] == [1, 1, None]  # Actual implementation behavior


def test_table_expr_fill_null_strategy_leading_nulls() -> None:
    """Test fill strategies with mixed patterns."""
    input_data = [None, None, 1, None, 2, None]
    expected_forward = [None, None, 1, 1, 2, 2]  # Actual implementation behavior
    expected_backward = [1, 1, 1, 2, 2, None]  # Actual implementation behavior

    daft_recordbatch = MicroPartition.from_pydict({"input": input_data})

    result_forward = daft_recordbatch.eval_expression_list([col("input").fill_null(strategy="forward")])
    assert result_forward.to_pydict()["input"] == expected_forward

    result_backward = daft_recordbatch.eval_expression_list([col("input").fill_null(strategy="backward")])
    assert result_backward.to_pydict()["input"] == expected_backward


def test_table_expr_fill_null_strategy_dtype_preservation() -> None:
    """Test that strategy-based fill_null preserves data types."""
    # Test with different numeric types
    test_cases = [
        ([None, 1, None], [None, 1, 1], "int64"),  # Forward fills from next valid value
        ([None, 1.5, None], [None, 1.5, 1.5], "float64"),
        ([None, True, None], [None, True, True], "bool"),
    ]

    for input_data, expected, dtype_name in test_cases:
        daft_recordbatch = MicroPartition.from_pydict({"input": input_data})
        result = daft_recordbatch.eval_expression_list([col("input").fill_null(strategy="forward")])
        result_data = result.to_pydict()["input"]

        assert result_data == expected
        # Ensure the non-null values maintain their original type
        if len([x for x in result_data if x is not None]) > 0:
            non_null_val = next(x for x in result_data if x is not None)
            if dtype_name == "int64":
                assert isinstance(non_null_val, int)
            elif dtype_name == "float64":
                assert isinstance(non_null_val, float)
            elif dtype_name == "bool":
                assert isinstance(non_null_val, bool)

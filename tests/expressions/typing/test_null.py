from __future__ import annotations

from daft.expressions import col
from tests.expressions.typing.conftest import (
    assert_typing_resolve_vs_runtime_behavior,
    has_supertype,
)


def test_is_null(unary_data_fixture):
    arg = unary_data_fixture
    assert_typing_resolve_vs_runtime_behavior(
        data=(unary_data_fixture,),
        expr=col(arg.name()).is_null(),
        run_kernel=lambda: arg.is_null(),
        resolvable=True,
    )


def test_not_null(unary_data_fixture):
    arg = unary_data_fixture
    assert_typing_resolve_vs_runtime_behavior(
        data=(unary_data_fixture,),
        expr=col(arg.name()).not_null(),
        run_kernel=lambda: arg.not_null(),
        resolvable=True,
    )


def test_fill_null(binary_data_fixture):
    lhs, rhs = binary_data_fixture
    assert_typing_resolve_vs_runtime_behavior(
        data=binary_data_fixture,
        expr=col(lhs.name()).fill_null(col(rhs.name())),
        run_kernel=lambda: lhs.fill_null(rhs),
        resolvable=has_supertype(lhs.datatype(), rhs.datatype()),
    )


def test_fill_null_strategy_forward(unary_data_fixture):
    arg = unary_data_fixture
    assert_typing_resolve_vs_runtime_behavior(
        data=(unary_data_fixture,),
        expr=col(arg.name()).fill_null(strategy="forward"),
        run_kernel=lambda: arg,  # Strategy preserves original dtype
        resolvable=True,
    )


def test_fill_null_strategy_backward(unary_data_fixture):
    arg = unary_data_fixture
    assert_typing_resolve_vs_runtime_behavior(
        data=(unary_data_fixture,),
        expr=col(arg.name()).fill_null(strategy="backward"),
        run_kernel=lambda: arg,  # Strategy preserves original dtype
        resolvable=True,
    )


def test_fill_null_strategy_vs_value_typing(binary_data_fixture):
    """Test that type resolution works correctly for both strategy and value-based fill_null."""
    lhs, rhs = binary_data_fixture

    # Test value-based fill_null (should resolve to supertype)
    assert_typing_resolve_vs_runtime_behavior(
        data=binary_data_fixture,
        expr=col(lhs.name()).fill_null(col(rhs.name())),
        run_kernel=lambda: lhs.fill_null(rhs),
        resolvable=has_supertype(lhs.datatype(), rhs.datatype()),
    )

    # Test strategy-based fill_null (should preserve original type)
    assert_typing_resolve_vs_runtime_behavior(
        data=(lhs,),
        expr=col(lhs.name()).fill_null(strategy="forward"),
        run_kernel=lambda: lhs,
        resolvable=True,
    )


def test_fill_null_strategy_type_preservation(unary_data_fixture):
    """Test that strategy-based fill_null preserves the exact input type."""
    arg = unary_data_fixture

    # Both forward and backward should preserve the input type exactly
    for strategy in ["forward", "backward"]:
        assert_typing_resolve_vs_runtime_behavior(
            data=(unary_data_fixture,),
            expr=col(arg.name()).fill_null(strategy=strategy),
            run_kernel=lambda: arg,  # Should return the exact same type
            resolvable=True,
        )

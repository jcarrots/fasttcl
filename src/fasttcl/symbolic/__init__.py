"""Exact symbolic compilation at any positive even TCL order.

This interface constructs symbolic coefficients; the fast numerical solver
in :mod:`fasttcl` continues to support orders two, four, and six.
"""

from .tcl_order import TCLOrder
from .tcl_reduction import TCLReductionError, TCLReductionResult, derive_tcl_reduction
from .tcl_scalarization import (
    ScalarCompilation, TCLScalarizationError, compile_scalar_terms,
    evaluate_scalar_terms,
)

__all__ = [
    "TCLOrder", "TCLReductionError", "TCLReductionResult", "derive_tcl_reduction",
    "ScalarCompilation", "TCLScalarizationError", "compile_scalar_terms",
    "evaluate_scalar_terms",
]

from typing import Dict, Optional
from ..operations.base import MathOperation
from ..operations.addition import Addition
from ..operations.subtraction import Subtraction
from numpy.typing import NDArray
from typing import Any

class MathEngine:
    """
    This class knows all operations and executes them.
    """

    def __init__(self):
        self._operations: Dict[str, MathOperation] = {}
        self._register_default_operations()

    def _register_default_operations(self) -> None:
        """Register to the factory all operations"""
        defaults = [
            Addition(),
            Subtraction(),
        ]
        for op in defaults:
            self.register(op)

    def register(self, operation: MathOperation) -> None:
        if operation.name in self._operations:
            raise ValueError(f"Operation '{operation.name}' already exists")
        self._operations[operation.name] = operation

    def get(self, name: str) -> Optional[MathOperation]:
        return self._operations.get(name)

    def execute(self, op_name: str, left: NDArray, right: Any) -> Any:
        op = self.get(op_name)
        if op is None:
            raise ValueError(f"Operation '{op_name}' does not supported")
        return op.execute(left, right)
    
    def add(self, left: NDArray, right: Any) -> NDArray:
        return self.execute("add", left, right)

    def sub(self, left: NDArray, right: Any) -> NDArray:
        return self.execute("sub", left, right)
"""Bounded AST interpreter for Strategist value heuristics, not an OS sandbox.

The restricted language never executes generated Python code. Game adapters must
supply public state and choose their game's payoff scale; the interpreter enforces
normalized expected net returns in [-1, 1] rather than clipping or rescaling them.
Only exact built-in JSON-like types (plus tuples) cross the interpreter boundary.
Explanations are string-keyed, bounded JSON-like dictionaries, not executable code.
"""

from __future__ import annotations

import ast
import hashlib
import math
import operator
import re


MAX_SOURCE = 12_000
MAX_AST_NODES = 2_000
MAX_AST_DEPTH = 60
MAX_FUEL = 10_000
MAX_ITEMS = 128
MAX_STRING = 512
MAX_DATA_NODES = 4_096
MAX_DATA_DEPTH = 32
MAX_CONSTANT = 1_000_000
MAX_NUMBER = 1_000_000_000
ZERO_SUM_TOLERANCE = 1e-8


class InvalidHeuristic(ValueError):
    """Source, input, execution, or output violates the restricted language."""


def _reject() -> None:
    # Never expose generated source, state values, or underlying exception text.
    raise InvalidHeuristic("Invalid or resource-limited value heuristic")


def _number(value: object, limit: int = MAX_NUMBER) -> bool:
    return (
        type(value) in (int, float)
        and -limit <= value <= limit
        and (type(value) is int or math.isfinite(value))
    )


def _data(value: object, *, copy: bool = False) -> object:
    """Validate before operating on objects; count keys and alias occurrences."""
    count = 0
    active: set[int] = set()

    def visit(item: object, depth: int) -> object:
        nonlocal count
        count += 1
        if count > MAX_DATA_NODES or depth > MAX_DATA_DEPTH:
            _reject()
        kind = type(item)
        if kind in (int, float):
            if not _number(item):
                _reject()
            return item
        if kind is str:
            if len(item) > MAX_STRING:
                _reject()
            return item
        if item is None or kind is bool:
            return item
        if kind not in (list, tuple, dict):
            _reject()
        if len(item) > MAX_ITEMS or id(item) in active:
            _reject()
        active.add(id(item))
        try:
            if kind is dict:
                result = {} if copy else None
                for key, child in item.items():
                    if type(key) is not str:
                        _reject()
                    visit(key, depth + 1)
                    checked = visit(child, depth + 1)
                    if copy:
                        result[key] = checked
            else:
                result = [] if copy else None
                for child in item:
                    checked = visit(child, depth + 1)
                    if copy:
                        result.append(checked)
                if copy and kind is tuple:
                    result = tuple(result)
        finally:
            active.remove(id(item))
        return result if copy else item

    return visit(value, 0)


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_ARITY = {
    "len": (1, 1), "sum": (1, 2), "min": (1, MAX_ITEMS),
    "max": (1, MAX_ITEMS), "abs": (1, 1), "float": (0, 1),
    "int": (0, 1), "round": (1, 2), "range": (1, 3),
}
_ALLOWED = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg,
    ast.Assign, ast.AugAssign, ast.If, ast.Return, ast.For,
    ast.Constant, ast.Name, ast.Subscript, ast.List, ast.Tuple, ast.Dict,
    ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp, ast.Call,
    ast.Load, ast.Store, ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or,
    ast.In, ast.NotIn,
    *_BINARY, *_COMPARE,
}


def _target(node: ast.AST) -> None:
    if (
        type(node) is not ast.Name
        or node.id in {"state", "evaluate_state", *_ARITY}
        or node.id.startswith("__")
    ):
        _reject()


def _validate(tree: ast.Module) -> ast.FunctionDef:
    if len(tree.body) != 1 or type(tree.body[0]) is not ast.FunctionDef:
        _reject()
    function = tree.body[0]
    args = function.args
    if (
        function.name != "evaluate_state" or function.decorator_list
        or function.returns is not None or function.type_comment is not None
        or getattr(function, "type_params", [])
        or args.posonlyargs or args.kwonlyargs or args.defaults
        or args.kw_defaults or args.vararg is not None or args.kwarg is not None
        or len(args.args) != 1 or args.args[0].arg != "state"
        or args.args[0].annotation is not None
        or args.args[0].type_comment is not None
    ):
        _reject()
    stack = [(tree, 0)]
    count = 0
    has_return = False
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_AST_NODES or depth > MAX_AST_DEPTH or type(node) not in _ALLOWED:
            _reject()
        if isinstance(node, ast.FunctionDef) and node is not function:
            _reject()
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            _reject()
        if isinstance(node, ast.Constant):
            value = node.value
            if type(value) in (int, float):
                if not _number(value, MAX_CONSTANT):
                    _reject()
            elif type(value) is str:
                if len(value) > MAX_STRING:
                    _reject()
            elif value is not None and type(value) is not bool:
                _reject()
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1 or node.type_comment is not None:
                _reject()
            _target(node.targets[0])
        elif isinstance(node, ast.AugAssign):
            _target(node.target)
        elif isinstance(node, ast.For):
            _target(node.target)
            if node.orelse or node.type_comment is not None:
                _reject()
        elif isinstance(node, ast.Return):
            has_return = True
            if node.value is None:
                _reject()
        elif isinstance(node, ast.Call):
            if type(node.func) is not ast.Name or node.func.id not in _ARITY or node.keywords:
                _reject()
            low, high = _ARITY[node.func.id]
            if not low <= len(node.args) <= high:
                _reject()
        elif isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > MAX_ITEMS:
                _reject()
        elif isinstance(node, ast.Dict):
            if len(node.keys) > MAX_ITEMS or any(key is None for key in node.keys):
                _reject()
            known = set()
            for key in node.keys:
                if isinstance(key, ast.Constant):
                    if type(key.value) is not str or key.value in known:
                        _reject()
                    known.add(key.value)
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    if not has_return:
        _reject()
    return function


class _Interpreter:
    def __init__(self, state: dict):
        self.env = {"state": state}
        self.fuel = MAX_FUEL

    def tick(self) -> None:
        self.fuel -= 1
        if self.fuel < 0:
            _reject()

    def bind(self, name: str, value: object) -> None:
        if name not in self.env and len(self.env) >= MAX_ITEMS:
            _reject()
        self.env[name] = value

    def binary(self, op: ast.operator, left: object, right: object) -> object:
        if type(op) is ast.Add and type(left) is type(right) and type(left) in (str, list, tuple):
            limit = MAX_STRING if type(left) is str else MAX_ITEMS
            if len(left) + len(right) > limit:
                _reject()
        elif not (_number(left) and _number(right)):
            # In particular, no sequence repetition or string interpolation.
            _reject()
        return _data(_BINARY[type(op)](left, right))

    def call(self, name: str, args: list) -> object:
        if name == "range":
            if any(type(arg) is not int for arg in args):
                _reject()
            values = range(*args)
            if len(values) > MAX_ITEMS:
                _reject()
            return list(values)
        if name == "len":
            if type(args[0]) not in (str, list, tuple, dict):
                _reject()
            return len(args[0])
        if name in ("sum", "min", "max"):
            if name == "sum" or len(args) == 1:
                values = args[0]
                if type(values) not in (list, tuple):
                    _reject()
            else:
                values = args
            if name != "sum" and not values:
                _reject()
            total = args[1] if name == "sum" and len(args) == 2 else 0
            if not _number(total):
                _reject()
            best = None
            for value in values:
                self.tick()
                if not _number(value):
                    _reject()
                if name == "sum":
                    total += value
                    if not _number(total):
                        _reject()
                elif best is None or (value < best if name == "min" else value > best):
                    best = value
            return total if name == "sum" else best
        if name in ("int", "float"):
            if args and not (_number(args[0]) or type(args[0]) is str):
                _reject()
            return (int if name == "int" else float)(*args)
        if not _number(args[0]):
            _reject()
        if name == "abs":
            return abs(args[0])
        if name == "round":
            if len(args) == 2 and (type(args[1]) is not int or not -15 <= args[1] <= 15):
                _reject()
            return round(*args)
        _reject()

    def expr(self, node: ast.expr) -> object:
        self.tick()
        return _data(self._expr(node))

    def _expr(self, node: ast.expr) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                _reject()
            return self.env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [self.expr(child) for child in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Dict):
            result = {}
            for key, value in zip(node.keys, node.values):
                key_value = self.expr(key)
                if type(key_value) is not str or key_value in result:
                    _reject()
                result[key_value] = self.expr(value)
            return result
        if isinstance(node, ast.Subscript):
            value, key = self.expr(node.value), self.expr(node.slice)
            if type(value) is dict:
                if type(key) is not str or key not in value:
                    _reject()
            elif type(value) in (str, list, tuple):
                if type(key) is not int or not -len(value) <= key < len(value):
                    _reject()
            else:
                _reject()
            return value[key]
        if isinstance(node, ast.BinOp):
            return self.binary(node.op, self.expr(node.left), self.expr(node.right))
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand)
            if isinstance(node.op, ast.Not):
                return not value
            if not _number(value):
                _reject()
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BoolOp):
            result = None
            for child in node.values:
                result = self.expr(child)
                if isinstance(node.op, ast.And) and not result:
                    break
                if isinstance(node.op, ast.Or) and result:
                    break
            return result
        if isinstance(node, ast.IfExp):
            return self.expr(node.body if self.expr(node.test) else node.orelse)
        if isinstance(node, ast.Compare):
            left = self.expr(node.left)
            for op, child in zip(node.ops, node.comparators):
                right = self.expr(child)
                if isinstance(op, (ast.In, ast.NotIn)):
                    if type(right) not in (str, list, tuple, dict):
                        _reject()
                    if type(right) in (str, dict) and type(left) is not str:
                        _reject()
                    matched = left in right
                    if isinstance(op, ast.NotIn):
                        matched = not matched
                else:
                    if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) and not (
                        (_number(left) and _number(right)) or type(left) is type(right) is str
                    ):
                        _reject()
                    matched = _COMPARE[type(op)](left, right)
                if not matched:
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            return self.call(node.func.id, [self.expr(arg) for arg in node.args])
        _reject()

    def block(self, statements: list[ast.stmt]) -> tuple[bool, object]:
        # An explicit return flag avoids executing user functions or exceptions.
        for node in statements:
            self.tick()
            if isinstance(node, ast.Return):
                return True, self.expr(node.value)
            if isinstance(node, ast.Assign):
                self.bind(node.targets[0].id, self.expr(node.value))
            elif isinstance(node, ast.AugAssign):
                name = node.target.id
                if name not in self.env:
                    _reject()
                self.bind(name, self.binary(node.op, self.env[name], self.expr(node.value)))
            elif isinstance(node, ast.If):
                returned, value = self.block(node.body if self.expr(node.test) else node.orelse)
                if returned:
                    return True, value
            elif isinstance(node, ast.For):
                values = self.expr(node.iter)
                if type(values) not in (list, tuple):
                    _reject()
                for item in values:
                    self.tick()
                    self.bind(node.target.id, item)
                    returned, value = self.block(node.body)
                    if returned:
                        return True, value
            else:
                _reject()
        return False, None


class ValueHeuristic:
    """Parse once, then interpret with an independent environment and fuel budget.

    No imports, attributes, mutation targets, slices, recursion, comprehensions,
    exceptions, decorators, annotations, keyword calls, or function-valued locals
    are supported. Arithmetic operands are real numbers, excluding booleans;
    bounded same-type sequence addition is also supported. range returns a list.
    min/max/sum accept bounded numeric lists/tuples; round's digits are in [-15,15].
    Input dictionaries require string keys. Explanations may contain strings,
    booleans, null, and nested lists/dicts/tuples as well as bounded finite numbers.
    Scores must be two real numbers (not bool), normalized and zero-sum to 1e-8.
    """

    def __init__(self, source: str):
        try:
            if type(source) is not str or len(source) > MAX_SOURCE:
                _reject()
            normalized = source.strip()
            if normalized.startswith("```"):
                match = re.fullmatch(r"```(?:python)?[ \t]*\r?\n(.*?)\r?\n```", normalized, re.DOTALL)
                if match is None or "```" in match.group(1):
                    _reject()
                normalized = match.group(1).strip()
            normalized = normalized.replace("\r\n", "\n").replace("\r", "\n") + "\n"
            if len(normalized) > MAX_SOURCE:
                _reject()
            tree = ast.parse(normalized, mode="exec", type_comments=True)
            function = _validate(tree)
            self._source = normalized
            self._sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            self._function = function
        except Exception:
            raise InvalidHeuristic("Invalid or resource-limited value heuristic") from None

    @property
    def source(self) -> str:
        return self._source

    @property
    def sha256(self) -> str:
        return self._sha256

    def evaluate(self, state: dict) -> tuple[tuple[float, float], dict]:
        try:
            if type(state) is not dict:
                _reject()
            snapshot = _data(state, copy=True)
            returned, result = _Interpreter(snapshot).block(self._function.body)
            if not returned or type(result) is not tuple or len(result) != 2:
                _reject()
            scores, explanation = result
            if type(scores) not in (list, tuple) or len(scores) != 2 or type(explanation) is not dict:
                _reject()
            if any(not _number(score, 1) for score in scores):
                _reject()
            first, second = float(scores[0]), float(scores[1])
            if abs(first + second) > ZERO_SUM_TOLERANCE:
                _reject()
            # Detach explanation aliases from the per-call environment.
            return (first, second), _data(explanation, copy=True)
        except Exception:
            raise InvalidHeuristic("Invalid or resource-limited value heuristic") from None

    def __call__(self, state: dict) -> tuple[tuple[float, float], dict]:
        return self.evaluate(state)
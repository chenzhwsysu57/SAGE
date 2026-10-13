"""Model-envelope extraction and bounded code dialect for fixed-budget training.

No exec/eval/import of generated code. Legacy value.py remains unchanged.
The raw model reply stays in the journal; extracted source is a derived artifact.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import re

from methods.strategist import value as legacy


class CodeOutputError(ValueError):
    def __init__(self, category, hint, line=None):
        self.diagnostic = {"category": category, "line": line, "hint": hint}
        super().__init__(f"{category}: {hint}" + (f" (line {line})" if line else ""))


def fail(category, hint, node=None):
    raise CodeOutputError(category, hint, getattr(node, "lineno", None))


def extract_source(response):
    if type(response) is not str or not response.strip():
        fail("empty_output", "Return a complete evaluate_state(state) function.")
    if len(response) > 24000:
        fail("reply_too_long", "Keep the whole reply below24000 characters and source below12000.")
    text = response.strip().replace("\r\n", "\n").replace("\r", "\n")
    # Try plain Python first: backticks inside valid string literals aren't fences.
    try:
        ast.parse(text)
    except SyntaxError:
        fences = list(re.finditer(r"(?m)^[ \t]*(`{3,}|~{3,})([^\n]*)$", text))
        if not fences:
            fail("python_syntax_or_prose", "Return plain Python or exactly one complete fenced Python block; no bare prose.")
        if len(fences) != 2:
            fail("ambiguous_or_incomplete_fence", "Provide exactly one complete code block, not multiple blocks or an unclosed block.")
        opening, closing = fences
        if (opening.group(1) != closing.group(1) or closing.group(2).strip()
                or opening.group(2).strip().lower() not in ("", "python", "py", "python3")):
            fail("invalid_fence", "Use one matching Python code fence.")
        text = text[opening.end():closing.start()].strip()
    if len(text) > legacy.MAX_SOURCE:
        fail("source_too_long", "Source must be <=12000 characters. Simplify the function.")
    try:
        tree = ast.parse(text, type_comments=True)
    except SyntaxError as exc:
        raise CodeOutputError("python_syntax", "Fix Python syntax; do not return partial code.", exc.lineno) from None
    if len(tree.body) != 1 or type(tree.body[0]) is not ast.FunctionDef:
        fail("function_envelope", "Return exactly one def evaluate_state(state), without top-level statements.")
    return text + "\n"


class _DialectValidation(ast.NodeTransformer):
    """Validate extensions by mapping to inert legacy AST while retaining operands.

    Only the ORIGINAL tree is interpreted. The surrogate is never executed and
    does not rewrite the model's policy semantics.
    """
    def __init__(self):
        self.loop_depth = 0

    def visit_For(self, node):
        self.loop_depth += 1
        result = self.generic_visit(node)
        self.loop_depth -= 1
        return result

    def visit_While(self, node):
        if node.orelse:
            fail("unsupported_while_else", "Use a while loop without else.", node)
        self.loop_depth += 1
        test = self.visit(node.test)
        body = [self.visit(n) for n in node.body]
        self.loop_depth -= 1
        return ast.copy_location(ast.For(target=ast.Name(id="dialect_control", ctx=ast.Store()),
            iter=ast.List(elts=[], ctx=ast.Load()), body=[ast.If(test=test, body=body, orelse=[])],
            orelse=[], type_comment=None), node)

    def visit_Continue(self, node):
        if not self.loop_depth:
            fail("control_outside_loop", "continue/break must be inside a loop.", node)
        return ast.copy_location(ast.Assign(targets=[ast.Name(id="dialect_control", ctx=ast.Store())],
                                          value=ast.Constant(value=0)), node)

    visit_Break = visit_Continue

    def visit_Expr(self, node):
        call = node.value
        if not (type(call) is ast.Call and type(call.func) is ast.Attribute
                and call.func.attr == "append" and type(call.func.value) is ast.Name
                and len(call.args) == 1 and not call.keywords):
            fail("unsupported_statement", "Only list.append(value) on a local list is supported as an expression statement.", node)
        return ast.copy_location(ast.Assign(targets=[ast.Name(id="dialect_append", ctx=ast.Store())],
            value=ast.List(elts=[self.visit(call.func.value), self.visit(call.args[0])], ctx=ast.Load())), node)

    def visit_Assign(self, node):
        if len(node.targets) == 1 and type(node.targets[0]) is ast.Subscript:
            target = node.targets[0]
            if type(target.value) is not ast.Name or target.value.id == "state" or node.type_comment:
                fail("unsupported_mutation_target", "Only local_list[index] or local_dict[string_key] assignment is supported; input is read-only.", node)
            return ast.copy_location(ast.Assign(targets=[ast.Name(id="dialect_store", ctx=ast.Store())],
                value=ast.List(elts=[self.visit(target.value), self.visit(target.slice), self.visit(node.value)],
                               ctx=ast.Load())), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        if type(node.func) is ast.Name and node.func.id == "str":
            if len(node.args) != 1 or node.keywords:
                fail("str_arity", "str accepts exactly one scalar argument.", node)
            return ast.copy_location(ast.Call(func=ast.Name(id="int", ctx=ast.Load()),
                args=[self.visit(node.args[0])], keywords=[]), node)
        return self.generic_visit(node)


def validate_tree(tree):
    stack = [(tree, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > legacy.MAX_AST_NODES or depth > legacy.MAX_AST_DEPTH:
            fail("ast_size", "Reduce code nesting or number of expressions.", node)
        if type(node) is ast.Name and type(node.ctx) is ast.Store and node.id == "str":
            fail("reserved_name", "Do not assign to builtin str.", node)
        stack.extend((n, depth+1) for n in ast.iter_child_nodes(node))
    surrogate = ast.fix_missing_locations(_DialectValidation().visit(copy.deepcopy(tree)))
    try:
        legacy._validate(surrogate)
    except legacy.InvalidHeuristic as exc:
        # Safe structural details, not arbitrary exception text or input contents.
        tb = exc.__traceback__
        rejected = None
        while tb:
            if tb.tb_frame.f_code.co_name == "_validate":
                rejected = tb.tb_frame.f_locals.get("node")
            tb = tb.tb_next
        name = type(rejected).__name__ if rejected is not None else "function"
        hint = f"Unsupported {name}. No imports, arbitrary attributes/methods, comprehensions, slices, nested functions, annotations or keyword calls."
        if isinstance(rejected, ast.Call) and isinstance(rejected.func, ast.Name):
            ident = rejected.func.id
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,40}", ident):
                hint = f"Unsupported call or arity: {ident}. Allowed builtins: " + ", ".join([*legacy._ARITY, "str"])
        fail("restricted_language", hint, rejected)


class _Continue(Exception):
    pass


class _Break(Exception):
    pass


class _CodeInterpreter(legacy._Interpreter):
    def __init__(self, state):
        super().__init__(state)
        self.line = None
        self.readonly = set()
        def mark(value):
            if type(value) in (list, tuple, dict):
                self.readonly.add(id(value))
                for item in (value.values() if type(value) is dict else value):
                    mark(item)
        mark(state)

    def expr(self, node):
        self.line = getattr(node, "lineno", self.line)
        try:
            return super().expr(node)
        except CodeOutputError:
            raise
        except Exception as exc:
            raise CodeOutputError("runtime_"+type(exc).__name__,
                "Check operand types, supported keys/index bounds, nonzero divisors and resource limits.",
                getattr(node, "lineno", self.line)) from None

    def call(self, name, args):
        if name == "str":
            if len(args) != 1 or type(args[0]) not in (str, int, float, bool, type(None)):
                fail("str_type", "str accepts a bounded JSON scalar, not a container/object.")
            return legacy._data(str(args[0]))
        return super().call(name, args)

    def _expr(self, node):
        if isinstance(node, ast.Subscript):
            value, key = self.expr(node.value), self.expr(node.slice)
            if type(value) is dict and type(key) is str and key not in value:
                label = key if re.fullmatch(r"[A-Za-z0-9_]{1,50}", key) else "[nonidentifier]"
                raise CodeOutputError("missing_key", f"Missing key '{label}'; use the supplied input schema or guard with 'in'.", node.lineno)
            if type(value) in (str, list, tuple) and (type(key) is not int or not -len(value) <= key < len(value)):
                fail("index_bounds", "Guard empty sequences and index bounds.", node)
            if type(value) is dict and type(key) is str or type(value) in (str, list, tuple) and type(key) is int:
                return value[key]
            fail("subscript_type", "Only string-keyed dictionaries and integer-indexed sequences are supported.", node)
        return super()._expr(node)

    def mutable(self, value, node):
        if type(value) not in (list, dict) or id(value) in self.readonly:
            fail("readonly_input", "Mutate only newly created local lists/dicts; observations and their aliases are read-only.", node)

    def block(self, statements):
        for node in statements:
            self.line = getattr(node, "lineno", None)
            self.tick()
            if isinstance(node, ast.Continue):
                raise _Continue()
            if isinstance(node, ast.Break):
                raise _Break()
            if isinstance(node, ast.Expr):
                call = node.value
                target = self.expr(call.func.value)
                self.mutable(target, node)
                if type(target) is not list or len(target) >= legacy.MAX_ITEMS:
                    fail("append_limit", "append requires a local list with fewer than128 items.", node)
                value = self.expr(call.args[0])
                legacy._data([*target, value])  # Reject cycles/oversize BEFORE mutation.
                if value is target:
                    fail("cyclic_container", "Do not append a list to itself.", node)
                target.append(value)
                legacy._data(target)
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
                place = node.targets[0]
                value = self.expr(node.value)
                target, key = self.expr(place.value), self.expr(place.slice)
                self.mutable(target, node)
                if type(target) is list:
                    if type(key) is not int or not -len(target) <= key < len(target):
                        fail("index_bounds", "Local assignment index is outside the list.", node)
                elif type(key) is not str or key not in target and len(target) >= legacy.MAX_ITEMS:
                    fail("dictionary_key", "Local dictionaries require string keys and <=128 entries.", node)
                target[key] = value
                legacy._data(target)
            elif isinstance(node, (ast.While, ast.For)):
                items = None
                if isinstance(node, ast.For):
                    items = self.expr(node.iter)
                    if type(items) not in (list, tuple):
                        fail("for_iterable", "for requires a bounded list/tuple or range().", node)
                    iterator = iter(items)
                while True:
                    self.tick()  # Bounds even while True: continue.
                    if isinstance(node, ast.While):
                        if not self.expr(node.test):
                            break
                    else:
                        try:
                            item = next(iterator)
                        except StopIteration:
                            break
                        self.bind(node.target.id, item)
                    try:
                        returned, value = self.block(node.body)
                    except _Continue:
                        continue
                    except _Break:
                        break
                    if returned:
                        return True, value
            else:
                returned, value = super().block([node])
                if returned:
                    return True, value
        return False, None


class FixedCodeHeuristic(legacy.ValueHeuristic):
    """Opt-in dialect; per-evaluation fuel/size limits and immutable input retained."""
    def __init__(self, source):
        normalized = extract_source(source)
        tree = ast.parse(normalized, type_comments=True)
        validate_tree(tree)
        self._source = normalized
        self._sha256 = hashlib.sha256(normalized.encode()).hexdigest()
        self._function = tree.body[0]

    def evaluate(self, state):
        if type(state) is not dict:
            fail("input_type", "Expected a structured observation dictionary.")
        interpreter = _CodeInterpreter(legacy._data(state, copy=True))
        try:
            returned, result = interpreter.block(self._function.body)
            if not returned or type(result) is not tuple or len(result) != 2:
                fail("return_schema", "Return ((v0,v1), explanation_dict).")
            scores, explanation = result
            if type(scores) not in (list, tuple) or len(scores) != 2 or type(explanation) is not dict:
                fail("return_schema", "Return exactly two numeric values and a string-keyed explanation dictionary.")
            if any(not legacy._number(v, 1) for v in scores) or abs(sum(scores)) > legacy.ZERO_SUM_TOLERANCE:
                fail("value_range", "v0,v1 must be finite in[-1,1], sum to zero and use fixed P0/P1 order.")
            return (float(scores[0]), float(scores[1])), legacy._data(explanation, copy=True)
        except CodeOutputError:
            raise
        except Exception as exc:
            raise CodeOutputError("runtime_"+type(exc).__name__,
                "Heuristic exceeded its resource limits or used an invalid operation; simplify loops and local data.",
                interpreter.line) from None


def output_contract(stage):
    if stage in ("seed_code", "code"):
        return {"format": "Exactly one complete def evaluate_state(state); plain Python preferred; one complete Python fence also accepted.",
            "return": "((v0,v1), explanation_dict), finite[-1,1], zero-sum, fixed P0/P1 order (NOT self/opponent order).",
            "limits": "Source<=12000 chars; reply<=24000; AST<=2000/depth60; fuel10000; containers<=128; strings<=512.",
            "supported": "if/elif/else, bounded for/range and fuel-limited while, break/continue, local list.append(x), local list[i]=x/dict[key]=x, scalar str(x), numeric arithmetic and comparisons.",
            "forbidden": "No imports, eval/exec, file/network access, arbitrary methods/attributes, list comprehension, slices, nested functions, annotations, keyword calls, input mutations or aliases thereof.",
            "guide": "Use concise code (aim<100 lines). Guard missing/None fields. Intermediate values are scalar or small JSON containers. Use only supplied observation keys; never invent private opponent state."}
    return {"format": "Plain text only, not code or a code block.",
            "max_characters": 4000 if stage == "idea" else 12000,
            "target": "One concrete idea in<=1000 characters; no full implementation." if stage == "idea" else "Concise reflection in<=3000 characters, citing supplied feedback only."}
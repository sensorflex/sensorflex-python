"""A library for operators"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
import types as _types_mod

from pydantic import BaseModel


# ============================================================
#  Helpers: Optional-aware, structural type compatibility
# ============================================================


def _strip_optional(t: Any) -> Tuple[Any, bool]:
    """
    If t is Optional[T] / Union[T, None] / (T | None), return (T, True).
    Otherwise (t, False).
    """
    origin = get_origin(t)

    # Handle typing.Union and PEP 604 unions (X | Y)
    if origin is Union or origin is _types_mod.UnionType:
        args = get_args(t)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(non_none) == len(args) - 1:
            return non_none[0], True

    return t, False


def _value_type_compatible(out_t: Any, in_t: Any) -> bool:
    """
    Check if a value of type `out_t` can be passed to a consumer expecting `in_t`.

    Rule:
      - Ignore Optional-ness when comparing (so Optional[int] <-> int is OK).
      - Then require base types to be equal or subclass-compatible.
    """
    out_base, _ = _strip_optional(out_t)
    in_base, _ = _strip_optional(in_t)

    if out_base == in_base:
        return True

    try:
        return issubclass(out_base, in_base)
    except TypeError:
        # not classes, might be typing constructs; be conservative
        return False


def _get_model_schema(cls: Type[Any]) -> Optional[Dict[str, Any]]:
    """
    If `cls` is a Pydantic model, return {field_name: field_type}.
    Otherwise, None.
    """
    if not isinstance(cls, type) or not issubclass(cls, BaseModel):
        return None

    # Pydantic models still use normal type hints for fields
    hints = get_type_hints(cls, include_extras=True)
    return hints


def _types_compatible(out_type: Type[Any], in_type: Type[Any]) -> bool:
    """
    Decide if out_type -> in_type is compatible.

    If both are Pydantic models, use a structural check:
      - out_type must provide *at least* all fields required by in_type
      - per-field types must be _value_type_compatible
        (so Optional[int] vs int is allowed)

    Otherwise, fall back to nominal check (same type or subclass).
    """
    out_schema = _get_model_schema(out_type)
    in_schema = _get_model_schema(in_type)

    if out_schema is not None and in_schema is not None:
        # Structural: out must provide everything in in
        for name, in_field_t in in_schema.items():
            if name not in out_schema:
                return False
            out_field_t = out_schema[name]
            if not _value_type_compatible(out_field_t, in_field_t):
                return False
        return True

    # Nominal fallback
    if out_type is in_type:
        return True
    try:
        return issubclass(out_type, in_type)
    except TypeError:
        return False


# ============================================================
#  Base Operator
# ============================================================


class Operator:
    """
    Base operator.

    Convention:
      Each *leaf* subclass MUST define two nested Pydantic models:

        class Foo(Operator):
            class Input(BaseModel):
                ...

            class Output(BaseModel):
                ...

    These are automatically picked up as the operator's
    input_type and output_type (no need for input_type=...).
    """

    input_type: Type[BaseModel]
    output_type: Type[BaseModel]

    def __init_subclass__(cls) -> None:
        # Auto-detect nested Pydantic models named Input/Output
        input_t: Optional[Type[BaseModel]] = None
        output_t: Optional[Type[BaseModel]] = None

        for name, obj in cls.__dict__.items():
            # narrow to class objects for static type checkers
            if not isinstance(obj, type):
                continue
            if not issubclass(obj, BaseModel):
                continue

            lname = name.lower()
            if lname in ("input", "inframe", "in", "framein"):
                input_t = obj
            elif lname in ("output", "outframe", "out", "frameout"):
                output_t = obj

        # Some combinator subclasses (Chain/Tap/ParAll/etc.) also define
        # dummy Input/Output; leaf ops define the real ones.
        if input_t is None or output_t is None:
            raise TypeError(
                f"{cls.__name__} must define nested Pydantic models named "
                f"'Input' and 'Output' (or InFrame/OutFrame)."
            )

        cls.input_type = input_t
        cls.output_type = output_t

    # --------------------------------------------------------

    def _coerce_input(self, v: Any) -> BaseModel:
        """
        Accept:
          - an instance of the expected Input model,
          - a dict (validated into the Input model),
          - another BaseModel whose type is structurally compatible
            with our Input type (e.g., X.Output -> Y.Input).
        """
        # 1) Already the right model
        if isinstance(v, self.input_type):
            return v

        # 2) Another Pydantic model: try structural conversion
        if isinstance(v, BaseModel):
            from_type = type(v)
            # use the same structural rule we used for chaining
            if _types_compatible(from_type, self.input_type):
                data = v.model_dump() if hasattr(v, "model_dump") else v.dict()
                if hasattr(self.input_type, "model_validate"):
                    return self.input_type.model_validate(data)  # pydantic v2
                return self.input_type.parse_obj(data)  # pydantic v1

            raise TypeError(
                f"{self.__class__.__name__} cannot accept BaseModel of type "
                f"{from_type.__name__}; incompatible with expected "
                f"{self.input_type.__name__}"
            )

        # 3) Dict input
        if isinstance(v, dict):
            if hasattr(self.input_type, "model_validate"):
                return self.input_type.model_validate(v)  # pydantic v2
            return self.input_type.parse_obj(v)  # pydantic v1

        # 4) Anything else: error
        raise TypeError(
            f"{self.__class__.__name__} expected {self.input_type.__name__}, "
            f"dict, or compatible BaseModel, got {type(v).__name__}"
        )

    def run(self, v: Any) -> Any:
        """
        Override in subclasses with the actual transformation.
        """
        raise NotImplementedError

    def __call__(self, v: Any) -> Any:
        coerced = self._coerce_input(v)
        out = self.run(coerced)
        return out

    def __or__(self, other: "Operator") -> "Chain":
        """
        Support X | Y chaining. Returns a Chain.
        """
        return Chain(self, other)

    def tap(self, *callbacks: Callable[[Any], None]) -> "Tap":
        """
        Returns a Tap wrapper that calls callbacks on the output.
        """
        return Tap(self, callbacks)


# ============================================================
#  Chain (sequential composition)
# ============================================================


class Chain(Operator):
    # Dummy Input/Output models to satisfy __init_subclass__
    class Input(BaseModel):
        pass

    class Output(BaseModel):
        pass

    def __init__(self, *ops: Operator) -> None:
        if not ops:
            raise ValueError("Chain needs at least one operator")

        # Check pairwise compatibility using our structural rule
        for a, b in zip(ops, ops[1:]):
            if not _types_compatible(a.output_type, b.input_type):
                raise TypeError(
                    "Incompatible chain:\n"
                    f"  {a.__class__.__name__}.Output = {a.output_type.__name__}\n"
                    f"  -> {b.__class__.__name__}.Input   = {b.input_type.__name__}"
                )

        self.ops: List[Operator] = list(ops)

        # Override instance-level input/output types with chain endpoints
        self.input_type = ops[0].input_type
        self.output_type = ops[-1].output_type

    def run(self, v: Any) -> Any:
        for op in self.ops:
            v = op(v)
        return v

    def __or__(self, other: Operator) -> "Chain":
        if isinstance(other, Chain):
            return Chain(*(self.ops + other.ops))
        return Chain(*(self.ops + [other]))


# ============================================================
#  Tap (side-effect wrapper)
# ============================================================


class Tap(Operator):
    class Input(BaseModel):
        pass

    class Output(BaseModel):
        pass

    def __init__(
        self, op: Operator, callbacks: Sequence[Callable[[Any], None]]
    ) -> None:
        self.op = op
        self.callbacks = list(callbacks)

        # Inherit I/O from wrapped operator
        self.input_type = op.input_type
        self.output_type = op.output_type

    def run(self, v: Any) -> Any:
        out = self.op(v)
        for cb in self.callbacks:
            cb(out)
        return out


# ============================================================
#  ParAll: run multiple ops with same input, collect all outputs
# ============================================================


class ParAllOp(Operator):
    class Input(BaseModel):
        pass

    class Output(BaseModel):
        results: Tuple[Any, ...]  # opaque tuple of outputs

    def __init__(self, *ops: Operator) -> None:
        if not ops:
            raise ValueError("ParAll needs at least one operator")

        first_in = ops[0].input_type
        for op in ops[1:]:
            if not _types_compatible(first_in, op.input_type):
                raise TypeError(
                    "ParAll: incompatible input types:\n"
                    f"  first  : {first_in.__name__}\n"
                    f"  other  : {op.input_type.__name__}"
                )

        self.ops: List[Operator] = list(ops)

        # Single input type; output is a Pydantic model wrapping a tuple
        self.input_type = first_in
        self.output_type = ParAllOp.Output

    def run(self, v: Any) -> ParAllOp.Output:
        results = tuple(op(v) for op in self.ops)
        return ParAllOp.Output(results=results)


def ParAll(*ops: Operator) -> ParAllOp:
    return ParAllOp(*ops)


# ============================================================
#  ParAny: run multiple ops with same input, pick first non-None
# ============================================================


class ParAnyOp(Operator):
    class Input(BaseModel):
        pass

    class Output(BaseModel):
        # Will just mirror the first operator's Output type at instance level
        pass

    def __init__(self, *ops: Operator) -> None:
        if not ops:
            raise ValueError("ParAny needs at least one operator")

        first_in = ops[0].input_type
        first_out = ops[0].output_type

        for op in ops[1:]:
            if not _types_compatible(first_in, op.input_type):
                raise TypeError(
                    "ParAny: incompatible input types:\n"
                    f"  first  : {first_in.__name__}\n"
                    f"  other  : {op.input_type.__name__}"
                )
            if not _types_compatible(op.output_type, first_out):
                raise TypeError(
                    "ParAny: incompatible output types:\n"
                    f"  first  : {first_out.__name__}\n"
                    f"  other  : {op.output_type.__name__}"
                )

        self.ops: List[Operator] = list(ops)
        self.input_type = first_in
        # output is "whatever the first op outputs"
        self.output_type = first_out

    def run(self, v: Any) -> Any:
        for op in self.ops:
            result = op(v)
            if result is not None:
                return result
        return None  # or raise


def ParAny(*ops: Operator) -> ParAnyOp:
    return ParAnyOp(*ops)


# ============================================================
#  Switch: conditional branching
# ============================================================


class SwitchOp(Operator):
    class Input(BaseModel):
        pass

    class Output(BaseModel):
        pass

    def __init__(
        self,
        branches: Sequence[Tuple[Callable[[Any], bool], Operator]],
        default: Optional[Operator] = None,
    ) -> None:
        if not branches and default is None:
            raise ValueError("Switch needs at least one branch or a default")

        all_ops: List[Operator] = [op for (_, op) in branches]
        if default is not None:
            all_ops.append(default)

        first_in = all_ops[0].input_type
        first_out = all_ops[0].output_type

        for op in all_ops[1:]:
            if not _types_compatible(first_in, op.input_type) or not _types_compatible(
                op.output_type, first_out
            ):
                raise TypeError(
                    f"Switch: incompatible branch operator {op.__class__.__name__}"
                )

        self.branches = list(branches)
        self.default = default

        self.input_type = first_in
        self.output_type = first_out

    def run(self, v: Any) -> Any:
        for predicate, op in self.branches:
            if predicate(v):
                return op(v)
        if self.default is not None:
            return self.default(v)
        raise ValueError("Switch: no branch matched and no default provided")


def switch(
    *branches: Tuple[Callable[[Any], bool], Operator],
    default: Optional[Operator] = None,
) -> SwitchOp:
    return SwitchOp(branches, default=default)

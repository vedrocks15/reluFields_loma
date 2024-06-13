"""
ASDL Module generated by asdl_adt
Original ASDL description:

module loma {
  func = FunctionDef ( string id, arg* args, stmt* body, bool is_simd, type? ret_type )
       | ForwardDiff ( string id, string primal_func )
       | ReverseDiff ( string id, string primal_func )
         attributes  ( int? lineno )

  stmt = Assign     ( expr target, expr val )
       | Declare    ( string target, type t, expr? val )
       | Return     ( expr val )
       | IfElse     ( expr cond, stmt* then_stmts, stmt* else_stmts )
       | While      ( expr cond, int max_iter, stmt* body )
       | CallStmt   ( expr call )
       attributes   ( int? lineno )

  expr = Var          ( string id )
       | ArrayAccess  ( expr array, expr index )
       | StructAccess ( expr struct, string member_id )
       | ConstFloat   ( float val )
       | ConstInt     ( int val )
       | BinaryOp     ( bin_op op, expr left, expr right )
       | Call         ( string id, expr* args )
       attributes     ( int? lineno, type? t )

  arg  = Arg ( string id, type t, inout i )

  type = Int    ( )
       | Float  ( )
       | Array  ( type t, int? static_size )
       | Struct ( string id, struct_member* members, int? lineno )
       | Diff   ( type t )

  struct_member = MemberDef ( string id, type t )

  bin_op = Add()
         | Sub()
         | Mul()
         | Div()
         | Less()
         | LessEqual()
         | Greater()
         | GreaterEqual()
         | Equal()
         | And()
         | Or()

  inout = In() | Out()
}

"""
from __future__ import annotations
import attrs as _attrs
from typing import Tuple as _Tuple
from typing import Optional as _Optional


def _list_to_tuple(xs):
    return tuple(xs) if isinstance(xs, list) else xs


class func:

    def __init__(self, *args, **kwargs):
        assert False, "Cannot instantiate func"


@_attrs.define(frozen=True)
class FunctionDef(func):
    id: str
    args: _Tuple[arg] = _attrs.field(converter=_list_to_tuple)
    body: _Tuple[stmt] = _attrs.field(converter=_list_to_tuple)
    is_simd: bool
    ret_type: _Optional[type] = None
    lineno: _Optional[int] = None

    def __new__(cls, id, args, body, is_simd, ret_type=None, lineno=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.id, str):
            raise TypeError("FunctionDef(...) argument 1: " +
                            "invalid instance of 'string id'")
        if not (isinstance(self.args, (tuple, list))
                and all(isinstance(x, arg) for x in self.args)):
            raise TypeError("FunctionDef(...) argument 2: " +
                            "invalid instance of 'arg* args'")
        if not (isinstance(self.body, (tuple, list))
                and all(isinstance(x, stmt) for x in self.body)):
            raise TypeError("FunctionDef(...) argument 3: " +
                            "invalid instance of 'stmt* body'")
        if not isinstance(self.is_simd, bool):
            raise TypeError("FunctionDef(...) argument 4: " +
                            "invalid instance of 'bool is_simd'")
        if not (self.ret_type is None or isinstance(self.ret_type, type)):
            raise TypeError("FunctionDef(...) argument 5: " +
                            "invalid instance of 'type? ret_type'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("FunctionDef(...) argument 6: " +
                            "invalid instance of 'int? lineno'")


@_attrs.define(frozen=True)
class ForwardDiff(func):
    id: str
    primal_func: str
    lineno: _Optional[int] = None

    def __new__(cls, id, primal_func, lineno=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.id, str):
            raise TypeError("ForwardDiff(...) argument 1: " +
                            "invalid instance of 'string id'")
        if not isinstance(self.primal_func, str):
            raise TypeError("ForwardDiff(...) argument 2: " +
                            "invalid instance of 'string primal_func'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("ForwardDiff(...) argument 3: " +
                            "invalid instance of 'int? lineno'")


@_attrs.define(frozen=True)
class ReverseDiff(func):
    id: str
    primal_func: str
    lineno: _Optional[int] = None

    def __new__(cls, id, primal_func, lineno=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.id, str):
            raise TypeError("ReverseDiff(...) argument 1: " +
                            "invalid instance of 'string id'")
        if not isinstance(self.primal_func, str):
            raise TypeError("ReverseDiff(...) argument 2: " +
                            "invalid instance of 'string primal_func'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("ReverseDiff(...) argument 3: " +
                            "invalid instance of 'int? lineno'")


class stmt:

    def __init__(self, *args, **kwargs):
        assert False, "Cannot instantiate stmt"


@_attrs.define(frozen=True)
class Assign(stmt):
    target: expr
    val: expr
    lineno: _Optional[int] = None

    def __new__(cls, target, val, lineno=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.target, expr):
            raise TypeError("Assign(...) argument 1: " +
                            "invalid instance of 'expr target'")
        if not isinstance(self.val, expr):
            raise TypeError("Assign(...) argument 2: " +
                            "invalid instance of 'expr val'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("Assign(...) argument 3: " +
                            "invalid instance of 'int? lineno'")


@_attrs.define(frozen=True)
class Declare(stmt):
    target: str
    t: type
    val: _Optional[expr] = None
    lineno: _Optional[int] = None

    def __new__(cls, target, t, val=None, lineno=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.target, str):
            raise TypeError("Declare(...) argument 1: " +
                            "invalid instance of 'string target'")
        if not isinstance(self.t, type):
            raise TypeError("Declare(...) argument 2: " +
                            "invalid instance of 'type t'")
        if not (self.val is None or isinstance(self.val, expr)):
            raise TypeError("Declare(...) argument 3: " +
                            "invalid instance of 'expr? val'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("Declare(...) argument 4: " +
                            "invalid instance of 'int? lineno'")


@_attrs.define(frozen=True)
class Return(stmt):
    val: expr
    lineno: _Optional[int] = None

    def __new__(cls, val, lineno=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.val, expr):
            raise TypeError("Return(...) argument 1: " +
                            "invalid instance of 'expr val'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("Return(...) argument 2: " +
                            "invalid instance of 'int? lineno'")


@_attrs.define(frozen=True)
class IfElse(stmt):
    cond: expr
    then_stmts: _Tuple[stmt] = _attrs.field(converter=_list_to_tuple)
    else_stmts: _Tuple[stmt] = _attrs.field(converter=_list_to_tuple)
    lineno: _Optional[int] = None

    def __new__(cls, cond, then_stmts, else_stmts, lineno=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.cond, expr):
            raise TypeError("IfElse(...) argument 1: " +
                            "invalid instance of 'expr cond'")
        if not (isinstance(self.then_stmts, (tuple, list))
                and all(isinstance(x, stmt) for x in self.then_stmts)):
            raise TypeError("IfElse(...) argument 2: " +
                            "invalid instance of 'stmt* then_stmts'")
        if not (isinstance(self.else_stmts, (tuple, list))
                and all(isinstance(x, stmt) for x in self.else_stmts)):
            raise TypeError("IfElse(...) argument 3: " +
                            "invalid instance of 'stmt* else_stmts'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("IfElse(...) argument 4: " +
                            "invalid instance of 'int? lineno'")


@_attrs.define(frozen=True)
class While(stmt):
    cond: expr
    max_iter: int
    body: _Tuple[stmt] = _attrs.field(converter=_list_to_tuple)
    lineno: _Optional[int] = None

    def __new__(cls, cond, max_iter, body, lineno=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.cond, expr):
            raise TypeError("While(...) argument 1: " +
                            "invalid instance of 'expr cond'")
        if not isinstance(self.max_iter, int):
            raise TypeError("While(...) argument 2: " +
                            "invalid instance of 'int max_iter'")
        if not (isinstance(self.body, (tuple, list))
                and all(isinstance(x, stmt) for x in self.body)):
            raise TypeError("While(...) argument 3: " +
                            "invalid instance of 'stmt* body'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("While(...) argument 4: " +
                            "invalid instance of 'int? lineno'")


@_attrs.define(frozen=True)
class CallStmt(stmt):
    call: expr
    lineno: _Optional[int] = None

    def __new__(cls, call, lineno=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.call, expr):
            raise TypeError("CallStmt(...) argument 1: " +
                            "invalid instance of 'expr call'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("CallStmt(...) argument 2: " +
                            "invalid instance of 'int? lineno'")


class expr:

    def __init__(self, *args, **kwargs):
        assert False, "Cannot instantiate expr"


@_attrs.define(frozen=True)
class Var(expr):
    id: str
    lineno: _Optional[int] = None
    t: _Optional[type] = None

    def __new__(cls, id, lineno=None, t=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.id, str):
            raise TypeError("Var(...) argument 1: " +
                            "invalid instance of 'string id'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("Var(...) argument 2: " +
                            "invalid instance of 'int? lineno'")
        if not (self.t is None or isinstance(self.t, type)):
            raise TypeError("Var(...) argument 3: " +
                            "invalid instance of 'type? t'")


@_attrs.define(frozen=True)
class ArrayAccess(expr):
    array: expr
    index: expr
    lineno: _Optional[int] = None
    t: _Optional[type] = None

    def __new__(cls, array, index, lineno=None, t=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.array, expr):
            raise TypeError("ArrayAccess(...) argument 1: " +
                            "invalid instance of 'expr array'")
        if not isinstance(self.index, expr):
            raise TypeError("ArrayAccess(...) argument 2: " +
                            "invalid instance of 'expr index'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("ArrayAccess(...) argument 3: " +
                            "invalid instance of 'int? lineno'")
        if not (self.t is None or isinstance(self.t, type)):
            raise TypeError("ArrayAccess(...) argument 4: " +
                            "invalid instance of 'type? t'")


@_attrs.define(frozen=True)
class StructAccess(expr):
    struct: expr
    member_id: str
    lineno: _Optional[int] = None
    t: _Optional[type] = None

    def __new__(cls, struct, member_id, lineno=None, t=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.struct, expr):
            raise TypeError("StructAccess(...) argument 1: " +
                            "invalid instance of 'expr struct'")
        if not isinstance(self.member_id, str):
            raise TypeError("StructAccess(...) argument 2: " +
                            "invalid instance of 'string member_id'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("StructAccess(...) argument 3: " +
                            "invalid instance of 'int? lineno'")
        if not (self.t is None or isinstance(self.t, type)):
            raise TypeError("StructAccess(...) argument 4: " +
                            "invalid instance of 'type? t'")


@_attrs.define(frozen=True)
class ConstFloat(expr):
    val: float
    lineno: _Optional[int] = None
    t: _Optional[type] = None

    def __new__(cls, val, lineno=None, t=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.val, float):
            raise TypeError("ConstFloat(...) argument 1: " +
                            "invalid instance of 'float val'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("ConstFloat(...) argument 2: " +
                            "invalid instance of 'int? lineno'")
        if not (self.t is None or isinstance(self.t, type)):
            raise TypeError("ConstFloat(...) argument 3: " +
                            "invalid instance of 'type? t'")


@_attrs.define(frozen=True)
class ConstInt(expr):
    val: int
    lineno: _Optional[int] = None
    t: _Optional[type] = None

    def __new__(cls, val, lineno=None, t=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.val, int):
            raise TypeError("ConstInt(...) argument 1: " +
                            "invalid instance of 'int val'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("ConstInt(...) argument 2: " +
                            "invalid instance of 'int? lineno'")
        if not (self.t is None or isinstance(self.t, type)):
            raise TypeError("ConstInt(...) argument 3: " +
                            "invalid instance of 'type? t'")


@_attrs.define(frozen=True)
class BinaryOp(expr):
    op: bin_op
    left: expr
    right: expr
    lineno: _Optional[int] = None
    t: _Optional[type] = None

    def __new__(cls, op, left, right, lineno=None, t=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.op, bin_op):
            raise TypeError("BinaryOp(...) argument 1: " +
                            "invalid instance of 'bin_op op'")
        if not isinstance(self.left, expr):
            raise TypeError("BinaryOp(...) argument 2: " +
                            "invalid instance of 'expr left'")
        if not isinstance(self.right, expr):
            raise TypeError("BinaryOp(...) argument 3: " +
                            "invalid instance of 'expr right'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("BinaryOp(...) argument 4: " +
                            "invalid instance of 'int? lineno'")
        if not (self.t is None or isinstance(self.t, type)):
            raise TypeError("BinaryOp(...) argument 5: " +
                            "invalid instance of 'type? t'")


@_attrs.define(frozen=True)
class Call(expr):
    id: str
    args: _Tuple[expr] = _attrs.field(converter=_list_to_tuple)
    lineno: _Optional[int] = None
    t: _Optional[type] = None

    def __new__(cls, id, args, lineno=None, t=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.id, str):
            raise TypeError("Call(...) argument 1: " +
                            "invalid instance of 'string id'")
        if not (isinstance(self.args, (tuple, list))
                and all(isinstance(x, expr) for x in self.args)):
            raise TypeError("Call(...) argument 2: " +
                            "invalid instance of 'expr* args'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("Call(...) argument 3: " +
                            "invalid instance of 'int? lineno'")
        if not (self.t is None or isinstance(self.t, type)):
            raise TypeError("Call(...) argument 4: " +
                            "invalid instance of 'type? t'")


class arg:

    def __init__(self, *args, **kwargs):
        assert False, "Cannot instantiate arg"


@_attrs.define(frozen=True)
class Arg(arg):
    id: str
    t: type
    i: inout

    def __new__(cls, id, t, i):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.id, str):
            raise TypeError("Arg(...) argument 1: " +
                            "invalid instance of 'string id'")
        if not isinstance(self.t, type):
            raise TypeError("Arg(...) argument 2: " +
                            "invalid instance of 'type t'")
        if not isinstance(self.i, inout):
            raise TypeError("Arg(...) argument 3: " +
                            "invalid instance of 'inout i'")


class type:

    def __init__(self, *args, **kwargs):
        assert False, "Cannot instantiate type"


@_attrs.define(frozen=True)
class Int(type):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass


@_attrs.define(frozen=True)
class Float(type):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass


@_attrs.define(frozen=True)
class Array(type):
    t: type
    static_size: _Optional[int] = None

    def __new__(cls, t, static_size=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.t, type):
            raise TypeError("Array(...) argument 1: " +
                            "invalid instance of 'type t'")
        if not (self.static_size is None or isinstance(self.static_size, int)):
            raise TypeError("Array(...) argument 2: " +
                            "invalid instance of 'int? static_size'")


@_attrs.define(frozen=True)
class Struct(type):
    id: str
    members: _Tuple[struct_member] = _attrs.field(converter=_list_to_tuple)
    lineno: _Optional[int] = None

    def __new__(cls, id, members, lineno=None):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.id, str):
            raise TypeError("Struct(...) argument 1: " +
                            "invalid instance of 'string id'")
        if not (isinstance(self.members, (tuple, list))
                and all(isinstance(x, struct_member) for x in self.members)):
            raise TypeError("Struct(...) argument 2: " +
                            "invalid instance of 'struct_member* members'")
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError("Struct(...) argument 3: " +
                            "invalid instance of 'int? lineno'")


@_attrs.define(frozen=True)
class Diff(type):
    t: type

    def __new__(cls, t):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.t, type):
            raise TypeError("Diff(...) argument 1: " +
                            "invalid instance of 'type t'")


class struct_member:

    def __init__(self, *args, **kwargs):
        assert False, "Cannot instantiate struct_member"


@_attrs.define(frozen=True)
class MemberDef(struct_member):
    id: str
    t: type

    def __new__(cls, id, t):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        if not isinstance(self.id, str):
            raise TypeError("MemberDef(...) argument 1: " +
                            "invalid instance of 'string id'")
        if not isinstance(self.t, type):
            raise TypeError("MemberDef(...) argument 2: " +
                            "invalid instance of 'type t'")


class bin_op:

    def __init__(self, *args, **kwargs):
        assert False, "Cannot instantiate bin_op"


@_attrs.define(frozen=True)
class Add(bin_op):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass


@_attrs.define(frozen=True)
class Sub(bin_op):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass


@_attrs.define(frozen=True)
class Mul(bin_op):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass


@_attrs.define(frozen=True)
class Div(bin_op):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass


@_attrs.define(frozen=True)
class Less(bin_op):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass


@_attrs.define(frozen=True)
class LessEqual(bin_op):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass


@_attrs.define(frozen=True)
class Greater(bin_op):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass


@_attrs.define(frozen=True)
class GreaterEqual(bin_op):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass


@_attrs.define(frozen=True)
class Equal(bin_op):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass


@_attrs.define(frozen=True)
class And(bin_op):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass


@_attrs.define(frozen=True)
class Or(bin_op):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass


class inout:

    def __init__(self, *args, **kwargs):
        assert False, "Cannot instantiate inout"


@_attrs.define(frozen=True)
class In(inout):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass


@_attrs.define(frozen=True)
class Out(inout):

    def __new__(cls):
        return super().__new__(cls)

    def __attrs_post_init__(self):
        pass
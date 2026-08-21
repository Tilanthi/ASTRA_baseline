# Copyright 2026 Glenn J. White
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Formal Logic Integration for STAN V40

Integrates:
- Z3 SMT Solver for constraint satisfaction
- Prolog-style inference rules
- Type theory for mathematical reasoning

Target: +15-20% on Math proofs and logical deduction

Date: 2025-12-11
Version: 40.0
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum
from abc import ABC, abstractmethod


class LogicType(Enum):
    """Types of logical reasoning"""
    PROPOSITIONAL = "propositional"
    FIRST_ORDER = "first_order"
    ARITHMETIC = "arithmetic"
    CONSTRAINT = "constraint"
    TYPE_THEORY = "type_theory"


class ProofStatus(Enum):
    """Status of a proof attempt"""
    UNKNOWN = "unknown"
    VALID = "valid"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class Constraint:
    """A logical constraint"""
    expression: str
    constraint_type: str  # equality, inequality, membership, etc.
    variables: List[str] = field(default_factory=list)
    domain: Optional[str] = None  # Int, Real, Bool, etc.

    def to_dict(self) -> Dict:
        return {
            'expression': self.expression,
            'type': self.constraint_type,
            'variables': self.variables,
            'domain': self.domain
        }


@dataclass
class ProofStep:
    """A step in a logical proof"""
    step_number: int
    statement: str
    justification: str
    rule_applied: str = ""
    dependencies: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'step': self.step_number,
            'statement': self.statement,
            'justification': self.justification,
            'rule': self.rule_applied,
            'deps': self.dependencies
        }


@dataclass
class LogicalProof:
    """A complete logical proof"""
    premises: List[str]
    conclusion: str
    steps: List[ProofStep] = field(default_factory=list)
    status: ProofStatus = ProofStatus.UNKNOWN
    logic_type: LogicType = LogicType.PROPOSITIONAL

    # Verification
    verified: bool = False
    counterexample: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            'premises': self.premises,
            'conclusion': self.conclusion,
            'steps': [s.to_dict() for s in self.steps],
            'status': self.status.value,
            'verified': self.verified
        }


class Z3Solver:
    """
    Z3 SMT Solver Interface.

    Provides constraint solving for:
    - Linear arithmetic
    - Boolean satisfiability
    - Array theory
    - Quantifiers

    Note: Uses pure Python fallback when Z3 not available.
    """

    def __init__(self):
        """Initialize Z3 solver."""
        self.solver = None
        try:
            import z3
            self.solver = z3.Solver()
            self.z3_available = True
        except ImportError:
            self.z3_available = False


    # ------------------------------------------------------------------ solving
    def solve(self, constraints: List[Constraint]) -> Dict[str, Any]:
        """
        Solve a constraint satisfaction problem.

        With z3 installed this is a full SMT call; the pure-Python fallback
        performs exhaustive boolean enumeration (exact) or bounded integer
        enumeration (declared in the result), and reports UNKNOWN outside
        those fragments.
        """
        if not constraints:
            return {"status": "sat", "model": {}, "method": "trivial"}

        if self.z3_available:
            return self._solve_with_z3(constraints)
        return self._solve_python_fallback(constraints)

    def _solve_with_z3(self, constraints: List[Constraint]) -> Dict[str, Any]:
        import z3
        solver = z3.Solver()
        var_map: Dict[str, Any] = {}

        def z3_var(name: str, domain: Optional[str]):
            if name not in var_map:
                kind = (domain or "").lower()
                if kind == "real":
                    var_map[name] = z3.Real(name)
                elif kind in ("bool", "boolean"):
                    var_map[name] = z3.Bool(name)
                else:
                    var_map[name] = z3.Int(name)
            return var_map[name]

        for constraint in constraints:
            expr = constraint.expression
            if constraint.variables:
                # Substitute declared variables; arithmetic stays symbolic
                safe = expr
                for name in constraint.variables:
                    safe = re.sub(rf"\b{re.escape(name)}\b",
                                  str(z3_var(name, constraint.domain)), safe)
                # Python comparison ops are z3 operator overloads
                try:
                    solver.add(eval(safe, {"__builtins__": {}},
                                    {str(v): v for v in var_map.values()}))
                    continue
                except Exception:
                    pass
            return {"status": "unknown",
                    "reason": f"could not translate: {expr!r}",
                    "method": "z3"}

        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            return {
                "status": "sat",
                "model": {str(d): str(model[d]) for d in model.decls()},
                "method": "z3",
            }
        if result == z3.unsat:
            return {"status": "unsat", "model": None, "method": "z3"}
        return {"status": "unknown", "model": None, "method": "z3"}

    # -------------------------------------------------- pure-python fallback
    BOOLEAN_KEYWORDS = {"and", "or", "not", "True", "False"}

    def _extract_vars(self, expression: str) -> List[str]:
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression))
        return sorted(tokens - self.BOOLEAN_KEYWORDS
                      - {"and", "or", "not", "implies", "iff"})

    def _split_top(self, expression: str, seps: Tuple[str, ...]) -> List[str]:
        """Split at top-level occurrences of any separator (parens respected)."""
        parts, current, depth = [], [], 0
        i = 0
        while i < len(expression):
            ch = expression[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0:
                matched = next((s for s in seps
                                if expression.startswith(s, i)), None)
                if matched:
                    parts.append("".join(current))
                    current = []
                    i += len(matched)
                    continue
            current.append(ch)
            i += 1
        parts.append("".join(current))
        return parts

    @staticmethod
    def _strip_parens(expr: str) -> str:
        """Remove ONE enclosing parenthesis layer, if it spans the whole expr."""
        s = expr.strip()
        if not (s.startswith("(") and s.endswith(")")):
            return s
        depth = 0
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    return s  # first pair closes early: not a single wrapper
        return s[1:-1]

    def _to_python_bool(self, expression: str) -> str:
        """Translate a propositional expression to fully parenthesised Python."""
        expr = expression
        # Word operators -> symbols first, so structural recursion sees them
        expr = re.sub(r"\biff\b|<=>", "<->", expr)
        expr = re.sub(r"\bimplies\b|=>", "->", expr)

        # Expose operators hidden under enclosing parentheses: recurse on
        # ((A -> B) -> C) by stripping the wrapper, then splitting below.
        while True:
            stripped = self._strip_parens(expr)
            if stripped == expr:
                break
            expr = stripped

        # <-> is the loosest binder: A <-> B  ==  (A -> B) and (B -> A)
        parts = [p for p in self._split_top(expr, ("<->",)) if p.strip()]
        if len(parts) > 1:
            left = self._to_python_bool(parts[0])
            right = self._to_python_bool("<->".join(parts[1:]))
            return (f"(((not ({left})) or ({right})) "
                    f"and ((not ({right})) or ({left})))")

        # -> is right-associative: A -> B -> C  ==  A -> (B -> C)
        parts = [p for p in self._split_top(expr, ("->",)) if p.strip()]
        if len(parts) > 1:
            left = self._to_python_bool(parts[0])
            right = self._to_python_bool("->".join(parts[1:]))
            return f"((not ({left})) or ({right}))"

        # Leaf: symbol-level operators
        expr = expr.replace("&", " and ").replace("|", " or ")
        expr = re.sub(r"!\s*|~\s*", " not ", expr)
        return f"({expr.strip()})"

    def _eval_bool(self, expression: str, assignment: Dict[str, bool]) -> bool:
        py = self._to_python_bool(expression)
        return bool(eval(py, {"__builtins__": {}}, dict(assignment)))

    def _solve_python_fallback(self, constraints: List[Constraint]) -> Dict[str, Any]:
        expressions = [c.expression for c in constraints]
        all_bool = all(
            not re.search(r"[0-9]", e) or not re.search(r"[<>]=?|==", e)
            for e in expressions
        )

        # Propositional fragment: exact truth-table enumeration
        if all_bool:
            variables: List[str] = []
            for e in expressions:
                for v in self._extract_vars(e):
                    if v not in variables:
                        variables.append(v)
            n = len(variables)
            if n <= 16:
                for bits in range(2 ** n):
                    assignment = {
                        v: bool((bits >> i) & 1)
                        for i, v in enumerate(variables)
                    }
                    if all(self._eval_bool(e, assignment) for e in expressions):
                        return {"status": "sat",
                                "model": {k: str(v) for k, v in assignment.items()},
                                "method": f"boolean-enumeration ({n} vars)"}
                return {"status": "unsat", "model": None,
                        "method": f"boolean-enumeration ({n} vars)"}

        # Arithmetic fragment: bounded integer enumeration (declared bounds)
        variables = sorted({v for e in expressions
                            for v in re.findall(r"[a-z][a-z0-9_]*", e)})
        if len(variables) <= 3:
            bounds = range(-10, 21)
            import itertools
            for values in itertools.product(bounds, repeat=len(variables)):
                assignment = dict(zip(variables, values))
                try:
                    if all(eval(e, {"__builtins__": {}}, assignment)
                           for e in expressions):
                        return {"status": "sat", "model": assignment,
                                "method": "bounded-int-enumeration",
                                "bounds": "each variable in [-10, 20]"}
                except Exception:
                    return {"status": "unknown",
                            "reason": "expression outside arithmetic fragment",
                            "method": "python-fallback"}
            return {"status": "unsat-in-bounds", "model": None,
                    "method": "bounded-int-enumeration",
                    "bounds": "each variable in [-10, 20]"}

        return {"status": "unknown",
                "reason": f"{len(variables)} variables exceeds fallback limit",
                "method": "python-fallback"}


class PrologEngine:
    """
    Minimal Prolog-style inference engine (pure Python).

    Supports Horn-clause programs with constants, variables (initial
    uppercase), and n-ary functors, solved by SLD resolution with
    unification, iterative deepening-free depth limit, and a proof trace.
    """

    MAX_DEPTH = 200
    MAX_SOLUTIONS = 50

    def __init__(self):
        self.clauses: List[Tuple[Any, List[Any]]] = []  # (head, body)

    # ------------------------------------------------------------------ program
    def add_fact(self, fact: str) -> None:
        self.clauses.append((self._parse_term(fact.strip().rstrip(".")), []))

    def add_rule(self, rule: str) -> None:
        """Add 'head :- body1, body2.' (facts may also be given bare)."""
        rule = rule.strip().rstrip(".")
        if ":-" in rule:
            head_str, body_str = rule.split(":-", 1)
            head = self._parse_term(head_str)
            body = [self._parse_term(t)
                    for t in self._split_args(body_str) if t.strip()]
        else:
            head, body = self._parse_term(rule), []
        self.clauses.append((head, body))

    def consult(self, program: str) -> None:
        """Load a multi-line program of clauses terminated by periods."""
        for statement in program.split("."):
            if statement.strip():
                self.add_rule(statement)

    # ------------------------------------------------------------------- terms
    def _parse_term(self, text: str):
        """Parse 'functor(a, X, b)' into (functor, [args])."""
        text = text.strip()
        match = re.fullmatch(r"([a-zA-Z_]\w*)\s*\((.*)\)", text, re.DOTALL)
        if match:
            functor, arg_str = match.group(1), match.group(2)
            args = [self._parse_term(a) for a in self._split_args(arg_str)]
            return (functor, args)
        if re.match(r"[A-Z_]", text):
            return ("__var__", text)          # variable
        return ("__const__", text)             # constant / atom

    @staticmethod
    def _split_args(arg_str: str) -> List[str]:
        parts, depth, current = [], 0, []
        for ch in arg_str:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append("".join(current)); current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return [p for p in (s.strip() for s in parts) if p]

    @staticmethod
    def _is_var(term) -> bool:
        return term[0] == "__var__"

    @staticmethod
    def _is_const(term) -> bool:
        return term[0] == "__const__"

    def _walk(self, term, subst):
        while self._is_var(term):
            name = term[1]
            if name in subst:
                term = subst[name]
            else:
                return term
        return term

    def _resolve(self, term, subst):
        term = self._walk(term, subst)
        if self._is_var(term) or self._is_const(term):
            return term
        functor, args = term
        return (functor, [self._resolve(a, subst) for a in args])

    def _unify(self, a, b, subst) -> Optional[Dict[str, Any]]:
        a, b = self._walk(a, subst), self._walk(b, subst)
        if self._is_var(a):
            return {**subst, a[1]: b}
        if self._is_var(b):
            return {**subst, b[1]: a}
        if self._is_const(a) or self._is_const(b):
            return subst if a == b else None
        if a[0] != b[0] or len(a[1]) != len(b[1]):
            return None
        for arg_a, arg_b in zip(a[1], b[1]):
            subst = self._unify(arg_a, arg_b, subst)
            if subst is None:
                return None
        return subst

    def _term_str(self, term, subst) -> str:
        term = self._resolve(term, subst)
        if self._is_var(term):
            return f"_G{term[1]}"
        functor, args = term
        if functor == "__const__":
            return str(args)
        if not args:
            return functor
        return f"{functor}({', '.join(self._term_str(a, subst) for a in args)})"

    # ------------------------------------------------------------------- query
    def query(self, query_str: str) -> Dict[str, Any]:
        """SLD-resolve a goal, returning bindings and a proof trace."""
        goal = self._parse_term(query_str.strip().rstrip("?.").strip())
        solutions: List[Dict[str, str]] = []
        trace: List[str] = []

        def resolve(goals, subst, depth):
            """
            SLD resolution by continuation: solve goals left-to-right,
            appending each clause body to the remaining goals.  A solution
            is emitted only when the goal list empties, so its substitution
            carries the complete binding chain for the original query.
            """
            if not goals:
                solutions.append(self._bindings(goal, subst))
                return True
            if (len(solutions) >= self.MAX_SOLUTIONS
                    or depth > self.MAX_DEPTH):
                return False

            selected = self._walk(goals[0], subst)
            rest = goals[1:]
            if self._is_var(selected):
                return False  # unresolved goal functor: no clause applies

            any_solution = False
            for head, body in self.clauses:
                r_head, r_body = self._rename_clause(head, body, depth)
                fresh = self._unify(selected, r_head, dict(subst))
                if fresh is None:
                    continue
                trace.append(f"{'fact' if not body else 'rule'}: "
                             f"{self._term_str(selected, fresh)}")
                if resolve(r_body + rest, fresh, depth + 1):
                    any_solution = True
            return any_solution

        resolve([goal], {}, 0)
        # Deduplicate identical binding sets
        seen, unique = set(), []
        for s in solutions:
            key = tuple(sorted(s.items()))
            if key not in seen:
                seen.add(key); unique.append(s)
        return {"solutions": unique[: self.MAX_SOLUTIONS],
                "n_solutions": len(unique), "proof_trace": trace[:50]}

    _rename_counter = 0

    def _rename_clause(self, head, body, tag):
        """
        Standardize apart: fresh variable names for one clause USE, with a
        single mapping shared by head and body so head-unification bindings
        constrain the body goals.
        """
        mapping: Dict[str, Any] = {}

        def walk(t):
            if self._is_var(t):
                if t[1] not in mapping:
                    PrologEngine._rename_counter += 1
                    mapping[t[1]] = ("__var__",
                                     f"{t[1]}_{tag}_{self._rename_counter}")
                return mapping[t[1]]
            if self._is_const(t):
                return t
            functor, args = t
            return (functor, [walk(a) for a in args])

        return walk(head), [walk(b) for b in body]

    def _bindings(self, goal, subst) -> Dict[str, str]:
        """Variable-name bindings of the original query goal."""
        result = {}
        def collect(term):
            if self._is_var(term):
                # Record the variable's ORIGINAL name -> fully resolved value
                resolved = self._resolve(term, subst)
                result[term[1]] = self._term_str(resolved, subst)
                return
            if self._is_const(term):
                return
            for arg in term[1]:
                collect(arg)
        collect(goal)
        return result


class FormalLogicEngine:
    """
    Facade over the formal reasoning backends.

    - Propositional entailment / tautology checking (exact, by model
      enumeration)
    - Constraint solving via Z3Solver
    - Prolog-style rule inference via PrologEngine
    """

    def __init__(self):
        self.z3 = Z3Solver()
        self.prolog = PrologEngine()

    # ------------------------------------------------------------ entailment
    def prove(self, premises: List[str], conclusion: str,
              logic_type: LogicType = LogicType.PROPOSITIONAL) -> LogicalProof:
        """
        Semantic entailment check: valid iff every assignment satisfying
        all premises also satisfies the conclusion (exact for <= 12 atoms).
        """
        variables: List[str] = []
        for expr in list(premises) + [conclusion]:
            for v in self.z3._extract_vars(expr):
                if v not in variables:
                    variables.append(v)
        n = len(variables)
        if n > 12:
            return LogicalProof(
                premises=premises, conclusion=conclusion,
                logic_type=logic_type, status=ProofStatus.UNKNOWN,
                steps=[ProofStep(1, f"{n} atomic propositions exceed "
                                    "enumeration limit (12)",
                                 "capacity check")],
            )

        steps = []
        counterexample = None
        for i, bits in enumerate(range(2 ** n)):
            assignment = {v: bool((bits >> k) & 1) for k, v in enumerate(variables)}
            if all(self.z3._eval_bool(p, assignment) for p in premises):
                if not self.z3._eval_bool(conclusion, assignment):
                    counterexample = {k: str(v) for k, v in assignment.items()}
                    steps.append(ProofStep(
                        i + 1,
                        f"Countermodel: {assignment}",
                        "premises true, conclusion false"))
                    break

        if counterexample is None:
            status = ProofStatus.VALID
            steps.insert(0, ProofStep(
                1, f"Checked all {2 ** n} assignments over {variables}",
                "exhaustive model enumeration"))
            steps.append(ProofStep(len(steps) + 1,
                                   f"Therefore {conclusion}",
                                   "universal entailment (modus ponens closure)"))
        else:
            status = ProofStatus.INVALID

        return LogicalProof(premises=premises, conclusion=conclusion,
                            steps=steps, status=status,
                            logic_type=logic_type,
                            verified=(status == ProofStatus.VALID),
                            counterexample=counterexample)

    def check_entailment(self, premises: List[str], conclusion: str) -> bool:
        return self.prove(premises, conclusion).status == ProofStatus.VALID

    def is_tautology(self, expression: str) -> bool:
        return self.check_entailment([], expression)

    # -------------------------------------------------------------- backends
    def solve_constraints(self, constraints: List[Constraint]) -> Dict[str, Any]:
        return self.z3.solve(constraints)

    def prolog_query(self, query: str) -> Dict[str, Any]:
        return self.prolog.query(query)

    # ------------------------------------------------------------- router API
    def solve(self, question: str) -> Tuple[Any, LogicalProof]:
        """
        Route a question to the supported formal fragments.

        Supported: pure arithmetic, 'P1, P2 |- C' entailment,
        'tautology: EXPR', 'solve: C1; C2' constraints. Anything else is
        honestly returned UNKNOWN.
        """
        q = question.strip()

        # Arithmetic
        if re.fullmatch(r"[-+*/%^(). \t0-9]+", q):
            try:
                value = eval(q.replace("^", "**"),
                             {"__builtins__": {}}, {})
                return value, LogicalProof(
                    premises=[], conclusion=q,
                    steps=[ProofStep(1, f"{q} = {value}", "arithmetic evaluation")],
                    status=ProofStatus.VALID,
                    logic_type=LogicType.ARITHMETIC, verified=True)
            except Exception:
                pass

        # Entailment: "P1, P2 |- C"  (also accepts 'therefore' spelling)
        if "|-" in q or "therefore" in q.lower():
            if "|-" in q:
                left, conclusion = q.split("|-", 1)
            else:
                left, conclusion = q.lower().split("therefore", 1)
            premises = [p for chunk in re.split(r"[,;]", left)
                        if (p := chunk.strip())]
            conclusion = conclusion.strip().rstrip(".")
            proof = self.prove(premises, conclusion)
            return (proof.status == ProofStatus.VALID,
                    proof) if premises else (None, LogicalProof(
                premises=[], conclusion=conclusion,
                status=ProofStatus.ERROR,
                steps=[ProofStep(1, "no premises parsed", "parse")]))

        # Tautology check
        taut = re.match(r"^(?:is\s+)?(.+?)\s+(?:a\s+)?tautology\??$",
                        q, re.IGNORECASE)
        if taut or q.lower().startswith("tautology:"):
            expr = (taut.group(1) if taut
                    else q.split(":", 1)[1]).strip()
            proof = self.prove([], expr)
            return (proof.status == ProofStatus.VALID, proof)

        # Constraint solving
        if q.lower().startswith("solve:"):
            rest = q.split(":", 1)[1]
            constraints = []
            for chunk in rest.split(";"):
                if chunk.strip():
                    constraints.append(Constraint(
                        expression=chunk.strip(),
                        constraint_type="assertion",
                        variables=self.z3._extract_vars(chunk)))
            result = self.z3.solve(constraints)
            status = ProofStatus.VALID if result.get("status") == "sat" \
                else ProofStatus.INVALID
            return result, LogicalProof(
                premises=[c.expression for c in constraints],
                conclusion=f"satisfiable: {result.get('status')}",
                steps=[ProofStep(1, str(result), result.get("method", ""))],
                status=status, logic_type=LogicType.CONSTRAINT)

        # Prolog: "prolog: QUERY" over previously consulted program
        if q.lower().startswith("prolog:"):
            return self.prolog.query(q.split(":", 1)[1]), LogicalProof(
                premises=[], conclusion=q,
                steps=[ProofStep(1, "SLD resolution", "prolog")],
                status=ProofStatus.VALID, logic_type=LogicType.FIRST_ORDER)

        return None, LogicalProof(
            premises=[], conclusion=question,
            steps=[ProofStep(1,
                             "outside supported fragments: arithmetic, "
                             "'P1, P2 |- C', 'tautology: EXPR', "
                             "'solve: C1; C2', 'prolog: QUERY'",
                             "routing")],
            status=ProofStatus.UNKNOWN)

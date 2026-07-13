"""snesrecomp.recompiler.v2.emit_function

Per-function v2 emit driver: decode_function → build_cfg → IR lowering
→ codegen → C function source.

Replaces v1's `emit_function()` (recomp.py:5930+, ~2200 lines including
EmitCtx state plumbing + heuristic phi machinery). The v2 driver is
explicit and stateless: each block is lowered independently, and
control flow between blocks is wired via labels + gotos based on the
v2 CFG edges.

Public API:
    emit_function(rom, bank, start, entry_m, entry_x, *, end=None,
                  func_name=None) -> str

Returns a complete `void <func_name>(CpuState *cpu) { ... }` C source
string.
"""

import sys
import pathlib

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_RECOMPILER_DIR = _THIS_DIR.parent
for p in (str(_THIS_DIR), str(_RECOMPILER_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from typing import Dict, List, Optional, Tuple  # noqa: E402

from v2.decoder import (  # noqa: E402
    DecodeKey, DecodedInsn, FunctionDecodeGraph, decode_function, addr24,
)
from v2.cfg import V2Block, V2CFG, build_cfg  # noqa: E402
from v2.lowering import lower  # noqa: E402
from v2.codegen import emit_op  # noqa: E402
from v2.ir import (  # noqa: E402
    IROp, IRBlock, Value,
    CondBranch, Goto, IndirectGoto, Call, Return,
    PullReg, PushReg, Pull, Push, Reg,
)


def _label_for(key: DecodeKey) -> str:
    """C label name for a block keyed by (pc, m, x)."""
    pc = key.pc & 0xFFFF
    return f"L_{pc:04X}_M{key.m}X{key.x}"


def _default_func_name(bank: int, start: int) -> str:
    return f"bank_{bank:02X}_{start:04X}"


def _variant_suffix(m: int, x: int) -> str:
    """Mirror of codegen._variant_suffix — kept duplicated to avoid the
    cross-module import cycle. Must stay in sync."""
    return f"_M{m & 1}X{x & 1}"


def _detect_garbage_variant(rom, bank, start, entry_m, entry_x, graph, end):
    """Return the PC of a split-immediate BRK if this (entry_m, entry_x) variant
    is a MISDECODE, else None.

    Signature: a `BRK` at PC P that a VALID sibling variant (opposite m OR
    opposite x) decodes as MID-INSTRUCTION — i.e. P is the high byte of a 16-bit
    immediate the narrow decode split (`LDA #$0007` at m=0 vs `LDA #$07`+`BRK` at
    m=1). Such a variant is never legitimately reached. Cross-variant comparison,
    decode-cached → cheap and conservative.

    Guards: only runs once the prune map exists (else a sibling might be an
    unreachable garbage decode); only compares against siblings that are actually
    VALID/reachable. Known limitation: a real `LDA #id; BRK` syscall reached at
    BOTH m would over-flag the m=1 variant — rare, and the runtime trap is
    non-fatal + deduped, so a stray hit is a cheap log, not a crash."""
    from v2.codegen import have_valid_variants, valid_variant_list
    if not have_valid_variants():
        return None
    brk_pcs = [k.pc & 0xFFFF for k, di in graph.insns.items()
               if di.insn.mnem == 'BRK']
    if not brk_pcs:
        return None
    valid = set(valid_variant_list(addr24(bank, start)))
    for sm, sx in ((entry_m ^ 1, entry_x), (entry_m, entry_x ^ 1)):
        if (sm & 1, sx & 1) not in valid:
            continue
        try:
            sg = decode_function(rom, bank, start, sm & 1, sx & 1, end=end)
        except Exception:
            continue
        for k2, di2 in sg.insns.items():
            q = k2.pc & 0xFFFF
            length = getattr(di2.insn, 'length', 0) or 0
            for p in brk_pcs:
                if q < p < q + length:      # P strictly inside a sibling insn
                    return p
    return None


def _find_equivalent_variants(rom, bank, start, entry_m, entry_x, graph,
                               end) -> List[Tuple[int, int]]:
    """Return the list of valid sibling (m, x) widths PROVEN equivalent
    to this (entry_m, entry_x) variant's decode, for the emit-truth
    prune's routing decision (codegen._route_pruned_variant).

    "Proven equivalent" means: every (pc16, mnemonic, operand length)
    this variant's decoder actually produced ALSO appears, identically,
    in the sibling's own decode at the same pc16. This is a coverage
    check, not a control-flow walk: if a function normalizes one flag
    (e.g. `REP #$20`) before any flag-sensitive instruction executes,
    every PC this variant reaches will decode identically regardless of
    the OTHER (unnormalized) flag's entry value, so the check passes
    vacuously true everywhere. If some flag-sensitive instruction (e.g.
    `LDX #imm`) decodes to a different operand length at a shared PC,
    that PC fails the check and the sibling is rejected outright — this
    is exactly the class of bug that misrouted $01:B898's pruned M1X0
    dispatch case to the wrong-width M1X1 body (2026-06-30/07-01): M1X1
    was never actually equivalent to M1X0, just assumed to be by a
    generic "nearest by (m,x) distance" heuristic with no proof behind
    it.

    Requires the reachable PC set to be non-trivial (more than just the
    shared entry point) so a degenerate/empty decode can't vacuously
    "match" everything — that would prove nothing.

    Deliberately does NOT gate on codegen.have_valid_variants() /
    valid_variant_list() the way the older _detect_garbage_variant does.
    Those reflect the emit-truth prune's CURRENT survivor set, which is
    empty on pass 0 and only gets populated as pruning proceeds — but a
    variant with a cfg-declared canonical (entry_mx:) can be pruned
    immediately after pass 0, before it's ever re-decoded at a later
    pass where that gate would be open. Gating this check the same way
    made it silently never fire for exactly that (fast, common) case —
    found via the $01:B898 regen report investigation (2026-07-01): the
    check proved correct in isolation but the pruned M1X0 case never
    appeared in cumulative_equivalences at all. Comparing against all 3
    other raw (m, x) combos unconditionally avoids the race: the
    equivalence fact needs to exist BEFORE prune/valid-variant state is
    settled, since that's what it's meant to inform.

    Cheap: decode_function results are cache-keyed by (rom identity,
    bank, start, m, x, end), so re-decoding a sibling already emitted
    elsewhere in this same pass is a cache hit, not a fresh decode.
    """
    this_shape: Dict[int, Tuple[str, int]] = {}
    for k, di in graph.insns.items():
        pc16 = k.pc & 0xFFFF
        length = getattr(di.insn, 'length', 0) or 0
        this_shape[pc16] = (di.insn.mnem, length)
    if len(this_shape) < 2:
        # Degenerate/near-empty decode (e.g. entry insn only) — no real
        # evidence either way. Don't claim equivalence.
        return []
    all_combos = ((0, 0), (0, 1), (1, 0), (1, 1))
    equivalents: List[Tuple[int, int]] = []
    for sm, sx in all_combos:
        if (sm, sx) == (entry_m & 1, entry_x & 1):
            continue
        try:
            sg = decode_function(rom, bank, start, sm & 1, sx & 1, end=end)
        except Exception:
            continue
        sib_shape: Dict[int, Tuple[str, int]] = {}
        for k2, di2 in sg.insns.items():
            pc16 = k2.pc & 0xFFFF
            length2 = getattr(di2.insn, 'length', 0) or 0
            sib_shape[pc16] = (di2.insn.mnem, length2)
        covered = True
        for pc16, shape in this_shape.items():
            if sib_shape.get(pc16) != shape:
                covered = False
                break
        if covered:
            equivalents.append((sm & 1, sx & 1))
    return equivalents


def _stack_width_for_a(insn) -> int:
    return 1 if (getattr(insn, 'm_flag', 1) & 1) else 2


def _stack_width_for_xy(insn) -> int:
    return 1 if (getattr(insn, 'x_flag', 1) & 1) else 2


def _stack_delta_for_trampoline_scan(di: DecodedInsn) -> Optional[int]:
    """Return local cpu->S delta for stack ops.

    Positive means bytes pushed below entry S; negative means bytes
    pulled. JSR/JSL/RTS/RTL stay zero because v2 uses the host C stack
    for call/return control. TCS/TXS overwrite S, so return None to
    reset the analysis state conservatively.
    """
    insn = di.insn
    mnem = insn.mnem
    if mnem in ('PEA', 'PEI', 'PER', 'PHD'):
        return 2
    if mnem in ('PHP', 'PHB', 'PHK'):
        return 1
    if mnem == 'PHA':
        return _stack_width_for_a(insn)
    if mnem in ('PHX', 'PHY'):
        return _stack_width_for_xy(insn)
    if mnem == 'PLD':
        return -2
    if mnem in ('PLP', 'PLB'):
        return -1
    if mnem == 'PLA':
        return -_stack_width_for_a(insn)
    if mnem in ('PLX', 'PLY'):
        return -_stack_width_for_xy(insn)
    if mnem in ('TCS', 'TXS'):
        return None
    return 0


def _clamp_stack_delta(value: int) -> int:
    # Keeps malformed loops from growing an unbounded lattice while
    # preserving every realistic local stack-frame delta seen in MMX.
    return max(-64, min(64, int(value)))


def _classify_trampoline_returns(cfg: V2CFG) -> set:
    """Find returns reached with unbalanced PEI/PEA/PER pushes.

    Each propagated state is (delta, saw_immediate_push). A Return is a
    candidate only when some path reaches it with non-zero delta and at
    least one PEI/PEA/PER on that path.
    """
    if cfg.entry not in cfg.blocks:
        return set()

    in_states: Dict[DecodeKey, set] = {cfg.entry: {(0, False)}}
    worklist: List[DecodeKey] = [cfg.entry]
    flagged: set = set()

    while worklist:
        key = worklist.pop()
        block = cfg.blocks.get(key)
        if block is None:
            continue
        states = set(in_states.get(key, set()))
        if not states:
            continue

        returned = False
        for di in block.insns:
            mnem = di.insn.mnem
            if mnem in ('RTS', 'RTL', 'RTI'):
                if any(delta != 0 and saw for delta, saw in states):
                    flagged.add(di.insn.addr & 0xFFFFFF)
                returned = True
                break

            delta = _stack_delta_for_trampoline_scan(di)
            saw_push = mnem in ('PEA', 'PEI', 'PER')
            next_states = set()
            for delta0, saw in states:
                if delta is None:
                    next_states.add((0, saw or saw_push))
                else:
                    next_states.add((
                        _clamp_stack_delta(delta0 + delta),
                        saw or saw_push,
                    ))
            states = next_states

        if returned:
            continue

        for succ in block.successors:
            if succ not in cfg.blocks:
                continue
            old = in_states.get(succ, set())
            new = old | states
            if new != old:
                in_states[succ] = new
                worklist.append(succ)

    return flagged


def scan_tail_call_stack_delta(
        rom: bytes, bank: int, start: int,
        entry_m: int, entry_x: int,
        *,
        end: Optional[int] = None,
        sibling_entry_pcs: Optional[set] = None,
        exclude_ranges=None,
        data_regions=None,
        dispatch_helpers=None,
        indirect_call_tables=None,
        indirect_dispatch=None,
        callee_exit_mx=None,
        callee_exit_mx_modes=None,
) -> Dict[int, int]:
    """Compute the net cpu->S delta at every same-bank tail-call exit.

    Decodes the function, runs a stack-height dataflow identical to
    _classify_trampoline_returns, and records the accumulated delta at
    each block exit whose successor PC is in `sibling_entry_pcs`.

    Returns {target_pc16: delta} where delta > 0 means net push (cpu->S
    decreased from entry — the typical PHB/PHK-then-PLB pattern).  The
    callee at target_pc16 should record `_entry_s = cpu->S + delta` so
    the host-return check passes after the callee's matching PLB epilogue.

    Only entries where ALL control-flow paths agree on the same non-zero
    integer delta are returned.  Indeterminate paths (TCS/TXS) or
    divergent paths are silently omitted — the cfg `entry_s_offset` hint
    remains available as a manual override for those cases.

    Returns {} on decode error, no tail-call exits, or no agreed delta.
    """
    if not sibling_entry_pcs:
        return {}

    try:
        graph = decode_function(
            rom, bank, start, entry_m, entry_x, end=end,
            dispatch_helpers=dispatch_helpers,
            indirect_call_tables=indirect_call_tables,
            indirect_dispatch=indirect_dispatch,
            data_regions=data_regions,
            callee_exit_mx=callee_exit_mx,
            callee_exit_mx_modes=callee_exit_mx_modes,
            sibling_entry_pcs=sibling_entry_pcs,
        )
    except Exception:
        return {}

    cfg = build_cfg(graph)
    if cfg.entry not in cfg.blocks:
        return {}

    # Dataflow: each state is either an int (net delta from entry) or None
    # (indeterminate — saw TCS/TXS).  We propagate sets of these values
    # along CFG edges, mirroring _classify_trampoline_returns.
    in_states: Dict[DecodeKey, set] = {cfg.entry: {0}}
    worklist: List[DecodeKey] = [cfg.entry]
    visited_edges: set = set()

    # target_pc16 -> set of observed deltas (None = indeterminate on >=1 path)
    exit_deltas: Dict[int, set] = {}

    while worklist:
        key = worklist.pop()
        block = cfg.blocks.get(key)
        if block is None:
            continue
        states = set(in_states.get(key, set()))
        if not states:
            continue

        for di in block.insns:
            if di.insn.mnem in ('RTS', 'RTL', 'RTI'):
                states = set()  # this path ends here; don't propagate
                break
            delta = _stack_delta_for_trampoline_scan(di)
            next_s = set()
            for s in states:
                if s is None or delta is None:
                    next_s.add(None)  # indeterminate
                else:
                    next_s.add(_clamp_stack_delta(s + delta))
            states = next_s

        if not states:
            continue

        for succ in block.successors:
            succ_pc16 = succ.pc & 0xFFFF
            if succ_pc16 in sibling_entry_pcs:
                # Tail-call exit into a sibling function
                s = exit_deltas.setdefault(succ_pc16, set())
                s |= states
            elif succ in cfg.blocks:
                edge = (key, succ)
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    old = in_states.get(succ, set())
                    merged = old | states
                    if merged != old:
                        in_states[succ] = merged
                        worklist.append(succ)

    result = {}
    for pc16, deltas in exit_deltas.items():
        if None in deltas:
            continue  # indeterminate path — skip; use manual entry_s_offset
        if len(deltas) != 1:
            continue  # divergent paths disagree — skip
        delta = next(iter(deltas))
        if delta != 0:
            result[pc16] = delta
    return result


def scan_rts_stack_deltas(
        rom: bytes, bank: int, start: int,
        entry_m: int, entry_x: int,
        *,
        end: Optional[int] = None,
        sibling_entry_pcs: Optional[set] = None,
        exclude_ranges=None,
        data_regions=None,
        dispatch_helpers=None,
        indirect_call_tables=None,
        indirect_dispatch=None,
        callee_exit_mx=None,
        callee_exit_mx_modes=None,
) -> Optional[int]:
    """Compute the net cpu->S delta from function entry to all RTS/RTL exits.

    Returns the single agreed-upon delta if ALL RTL/RTS exits in the
    function observe the same deterministic stack height, or None if
    any exit is indeterminate (TCS/TXS) or the exits disagree.

    delta > 0 means net push (cpu->S decreased from entry — e.g. PHB at
    entry that isn't matched by PLB before RTL).
    delta < 0 means net pop  (cpu->S increased from entry — e.g. PLB
    before RTL restoring a PHB pushed by the tail-call caller).
    delta = 0 means stack-balanced (standard case).
    """
    try:
        graph = decode_function(
            rom, bank, start, entry_m, entry_x, end=end,
            dispatch_helpers=dispatch_helpers,
            indirect_call_tables=indirect_call_tables,
            indirect_dispatch=indirect_dispatch,
            data_regions=data_regions,
            callee_exit_mx=callee_exit_mx,
            callee_exit_mx_modes=callee_exit_mx_modes,
            sibling_entry_pcs=sibling_entry_pcs or set(),
        )
    except Exception:
        return None

    cfg = build_cfg(graph)
    if cfg.entry not in cfg.blocks:
        return None

    in_states: Dict[DecodeKey, set] = {cfg.entry: {0}}
    worklist: List[DecodeKey] = [cfg.entry]
    visited_edges: set = set()
    rts_deltas: set = set()

    while worklist:
        key = worklist.pop()
        block = cfg.blocks.get(key)
        if block is None:
            continue
        states = set(in_states.get(key, set()))
        if not states:
            continue

        for di in block.insns:
            if di.insn.mnem in ('RTS', 'RTL', 'RTI'):
                rts_deltas |= states
                states = set()
                break
            delta = _stack_delta_for_trampoline_scan(di)
            next_s = set()
            for s in states:
                if s is None or delta is None:
                    next_s.add(None)
                else:
                    next_s.add(_clamp_stack_delta(s + delta))
            states = next_s

        if not states:
            continue

        for succ in block.successors:
            if succ in cfg.blocks:
                edge = (key, succ)
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    old = in_states.get(succ, set())
                    merged = old | states
                    if merged != old:
                        in_states[succ] = merged
                        worklist.append(succ)

    if not rts_deltas:
        return None
    if None in rts_deltas:
        return None
    if len(rts_deltas) != 1:
        return None
    return next(iter(rts_deltas))


def emit_function(rom: bytes, bank: int, start: int,
                  entry_m: int, entry_x: int,
                  *, end: Optional[int] = None,
                  func_name: Optional[str] = None,
                  dispatch_helpers=None,
                  indirect_call_tables=None,
                  indirect_dispatch=None,
                  suppressed_collector=None,
                  const_z_fold_collector=None,
                  dispatch_target_suppressed_collector=None,
                  unresolved_indirect_collector=None,
                  equivalence_collector=None,
                  data_regions=None,
                  exclude_ranges: Optional[List[Tuple[int, int]]] = None,
                  tail_call_pc16: Optional[int] = None,
                  tail_call_target_name: Optional[str] = None,
                  callee_exit_mx=None,
                  callee_exit_mx_modes=None,
                  sibling_entry_pcs: Optional[set] = None,
                  hle_spc_upload=None,
                  hle_func=None,
                  hle_dispatch=None,
                  entry_s_offset: int = 0) -> str:
    """Emit a complete v2 C function source for one 65816 function.

    Pipeline:
        rom + (bank, start, entry_m, entry_x) → decode_function
                                              → build_cfg
        for each block:
            lower(insn) → IR ops
            emit_op(op)  → C lines
        block-end op → goto / fall-through wiring
    """
    base_func_name = func_name if func_name is not None else _default_func_name(bank, start)
    # HLE bypass: if cfg declared this function's PC as the SPC upload
    # entry (`hle_spc_upload <pc>` in the bank cfg), replace the entire
    # decoded body with a single RtlUploadSpcImageFromDp call. The
    # standard SNES SPC upload protocol is a length/target/data block
    # stream pointed to by a 24-bit ROM pointer in direct page; the
    # runtime walks it directly and writes into apu->ram. Game-agnostic
    # — works for any project that uses the standard protocol (verified
    # on SMW's HandleSPCUploads_Inner $00:8079 and ALttP's LoadSongBank
    # $00:8888).
    if hle_spc_upload and (start & 0xFFFF) in set(hle_spc_upload):
        variant_name = f"{base_func_name}{_variant_suffix(entry_m, entry_x)}"
        pc24 = ((bank & 0xFF) << 16) | (start & 0xFFFF)
        return "\n".join([
            f"RecompReturn {variant_name}(CpuState *cpu) {{",
            "  extern const char *g_last_recomp_func;",
            "  extern bool RtlUploadSpcImageFromDp(CpuState *cpu);",
            f"  g_last_recomp_func = \"{variant_name}\";",
            f"  RecompStackPush(\"{variant_name}\");",
            f"  cpu_dbg_funcname(\"{variant_name}\");",
            f"  cpu_trace_func_entry(cpu, 0x{pc24:06X}, \"{variant_name}\");",
            f"  cpu_trace_block(cpu, 0x{pc24:06X});",
            "  WatchdogCheck();",
            "  if (!RtlUploadSpcImageFromDp(cpu)) {",
            f"    fprintf(stderr, \"[apu] {base_func_name} HLE upload failed\\n\");",
            "  }",
            "  RecompStackPop();",
            "  return RECOMP_RETURN_NORMAL;",
            "}",
            "",
        ])

    # Generic HLE: cfg declared `hle_func <pc> <c_helper>`. The forwarding
    # stub is emitted INSTEAD of the decoded body — but the normal
    # decode+codegen below still runs and its text is discarded at the final
    # return. Emission is what registers a body's call/goto targets with the
    # post-emit auto-promote pass, so replacing a function's body must not
    # orphan callees reachable only through it (ActRaiser $00:923A, reachable
    # only from the hle'd $00:8C98, silently stopped being emitted and broke
    # the link). Register-everything-first, THEN substitute the stub.
    hle_stub_src = None
    if hle_func and (start & 0xFFFF) in hle_func:
        c_helper = hle_func[start & 0xFFFF]
        variant_name = f"{base_func_name}{_variant_suffix(entry_m, entry_x)}"
        pc24 = ((bank & 0xFF) << 16) | (start & 0xFFFF)
        hle_stub_src = "\n".join([
            f"RecompReturn {variant_name}(CpuState *cpu) {{",
            "  extern const char *g_last_recomp_func;",
            f"  extern RecompReturn {c_helper}(CpuState *cpu);",
            f"  g_last_recomp_func = \"{variant_name}\";",
            f"  RecompStackPush(\"{variant_name}\");",
            f"  cpu_dbg_funcname(\"{variant_name}\");",
            f"  cpu_trace_func_entry(cpu, 0x{pc24:06X}, \"{variant_name}\");",
            f"  cpu_trace_block(cpu, 0x{pc24:06X});",
            "  WatchdogCheck();",
            f"  RecompReturn _r = {c_helper}(cpu);",
            "  RecompStackPop();",
            "  return _r;",
            "}",
            "",
        ])

    try:
        graph = decode_function(rom, bank, start, entry_m, entry_x, end=end,
                            dispatch_helpers=dispatch_helpers,
                            indirect_call_tables=indirect_call_tables,
                            indirect_dispatch=indirect_dispatch,
                            hle_dispatch=hle_dispatch,
                            data_regions=data_regions,
                            callee_exit_mx=callee_exit_mx,
                            callee_exit_mx_modes=callee_exit_mx_modes,
                            sibling_entry_pcs=sibling_entry_pcs)
    except Exception:
        # An hle'd body is allowed to be undecodable (that can be exactly why
        # it was hle'd). Fall back to the stub alone — discovery through this
        # body is then impossible, matching the pre-fix behavior.
        if hle_stub_src is not None:
            return hle_stub_src
        raise
    # Garbage-variant detection: a split-immediate MISDECODE. If this variant
    # decodes a BRK at a PC that a VALID sibling variant (opposite m or x) spans
    # mid-instruction, the BRK is the high byte of a 16-bit immediate the narrow
    # decode split off (e.g. m=0 `LDA #$0007` -> m=1 `LDA #$07`+`BRK`). Such a
    # variant is never legitimately reached; the runtime trap (emitted in the
    # prologue below) fires the instant a leaked flag dispatches into it — closer
    # to the misdecode root than the downstream crash, no oracle needed.
    _garbage_brk_pc = _detect_garbage_variant(
        rom, bank, start, entry_m, entry_x, graph, end)
    # Proven-equivalence check (2026-07-01): for every valid sibling
    # width, prove (or refute) that it's a safe substitute for THIS
    # variant via a full instruction-shape coverage check (see
    # _find_equivalent_variants docstring). Feeds codegen's
    # _route_pruned_variant so a pruned dispatch case routes to a
    # sibling PROVEN equivalent, not one merely assumed to be by the
    # generic (m,x)-distance heuristic.
    if equivalence_collector is not None:
        func_entry_pc24 = addr24(bank, start)
        for (sm, sx) in _find_equivalent_variants(
                rom, bank, start, entry_m, entry_x, graph, end):
            equivalence_collector.append(
                (func_entry_pc24, entry_m & 1, entry_x & 1, sm, sx))
    # Forward any suppressed indirect calls upward so emit_bank can
    # aggregate them into the build report. List-of-records.
    if suppressed_collector is not None:
        suppressed_collector.extend(graph.suppressed_indirect_calls)
    # Same plumbing for the constant-Z fold log: each rewritten BEQ/BNE
    # is recorded once per (function entry, branch site).
    if const_z_fold_collector is not None:
        const_z_fold_collector.extend(graph.const_z_folds)
    # cfg-data_region dispatch-target suppressions.
    if dispatch_target_suppressed_collector is not None:
        dispatch_target_suppressed_collector.extend(
            graph.dispatch_targets_suppressed)
    # IndirectGoto / Call (abs,X) sites that couldn't be resolved (no
    # cfg directive, no auto-recovery). v2_regen hard-fails on any.
    if unresolved_indirect_collector is not None:
        unresolved_indirect_collector.extend(graph.unresolved_indirects)
    cfg = build_cfg(graph)

    # ── PEI-trampoline detector (2026-05-24, narrow variant) ──────────
    #
    # Scan the function's decoded insns for PEI / PEA / PER mnemonics.
    # Each is a 2-byte push that doesn't have a matching pop in the
    # canonical asm-balanced shape (PHP/PLP, PHA/PLA, etc.). A function
    # containing any of them is a CANDIDATE PEI-trampoline — at its
    # RTS/RTL, the topmost cpu->S bytes may be a computed return target
    # rather than the caller's pushed JSR/JSL frame.
    #
    # Flagging is per-function: every Return in a candidate function
    # gets `source_pc24 ∈ _TRAMPOLINE_RETURNS`. Codegen.py's _emit_return
    # then emits a runtime balance check (cpu->S vs _entry_s) and on
    # the unbalanced path tail-calls cpu_dispatch_pc with the popped
    # (PB:PC+1) target. The runtime check filters out balanced paths
    # through the same Return op (e.g. a join block reached from both
    # a PEI-pushing predecessor and a non-PEI predecessor).
    #
    # PHP/PLP-balanced functions don't trip the detector (no PEI/PEA/PER)
    # so the standard `return _ps;` emit is used everywhere. The Dr Light
    # case (bank_04_9A02 with 3 PEIs on path C) is caught.
    has_pei = False
    for blk in cfg.blocks.values():
        for di in blk.insns:
            mnem = di.insn.mnem
            if mnem in ('PEI', 'PEA', 'PER'):
                has_pei = True
                break
        if has_pei:
            break
    has_pei = False
    trampoline_returns_local = _classify_trampoline_returns(cfg)
    if has_pei:
        # Every Return in this function may execute on an unbalanced path
        # (we don't do path-level discrimination — the runtime check is
        # the final discriminator). Collect every RTS/RTL/RTI source_pc24
        # and add to the codegen-level set.
        for blk in cfg.blocks.values():
            last = blk.insns[-1] if blk.insns else None
            if last is None:
                continue
            m = last.insn.mnem
            if m in ('RTS', 'RTL', 'RTI'):
                trampoline_returns_local.add(last.insn.addr & 0xFFFFFF)
        from v2.codegen import add_trampoline_returns
        add_trampoline_returns(trampoline_returns_local)
    if trampoline_returns_local:
        from v2.codegen import add_trampoline_returns
        add_trampoline_returns(trampoline_returns_local)

    if func_name is None:
        func_name = _default_func_name(bank, start)
    # Always append the (m, x) variant suffix. A 65816 function reached
    # from contexts with different (m, x) is literally a different
    # instruction stream (LDA/LDX/LDY immediate widths change), so each
    # variant gets its own C body. Hand-written entry points (I_RESET,
    # I_NMI, etc.) call into the cfg-default variant via aliases that
    # emit_bank produces.
    func_name = f"{func_name}{_variant_suffix(entry_m, entry_x)}"

    # Exit-invariant check context (AR_EXITMX / AR_EXITS): tell codegen this
    # function's name + the exit (m,x) recorded for its callers, so _emit_return
    # can validate it at every RTS/RTL. The recorded exit-mx is the callee_exit_mx
    # value THIS variant advertised — a runtime mismatch is the caller-poisoning
    # exit-mx misdecode class ($03:9156). None when the analyzer left it
    # ambiguous (callers "preserve") -> exit-mx check skipped for this variant.
    _fn_pc24 = (bank << 16) | (start & 0xFFFF)
    _rec_exit_mx = (callee_exit_mx or {}).get(
        (_fn_pc24, entry_m & 1, entry_x & 1))
    from v2.codegen import set_current_exit_ctx
    set_current_exit_ctx(func_name, _rec_exit_mx)

    # Mint a per-function value-id counter shared across all blocks.
    counter = [0]
    def vf():
        counter[0] += 1
        return Value(vid=counter[0])

    # Lower every (insn, key) into its IR ops, in block order.
    block_lines: Dict[DecodeKey, List[str]] = {}
    block_order: List[DecodeKey] = []

    # Emit blocks in a stable order: entry first, then DFS over successors.
    visited = set()
    def order_blocks(k):
        if k in visited or k not in cfg.blocks:
            return
        visited.add(k)
        block_order.append(k)
        for s in cfg.blocks[k].successors:
            order_blocks(s)
    order_blocks(cfg.entry)

    # Any cfg block not reached via DFS (defensive — shouldn't happen for
    # a well-formed v2 graph) gets appended at the end.
    for k in cfg.blocks:
        if k not in visited:
            block_order.append(k)

    # Set of labels that actually correspond to blocks in this function.
    # Successors that don't resolve to a local block (cross-bank, past `end:`,
    # indirect dispatch with unknown table) get tail-called as a separate
    # function. The auto-promote loop in v2_regen ensures that callable
    # function exists in the same emit pass (or a later iteration).
    local_labels = {_label_for(k) for k in block_order}
    # Push the label set to codegen so _emit_indirect_dispatch's ret-
    # continuation can decide goto-vs-tailcall by what the decoder actually
    # created in THIS function (see codegen._CUR_LOCAL_LABELS).
    from v2.codegen import set_current_local_labels
    set_current_local_labels(local_labels)

    # ── Pre-lower IR for every block ────────────────────────────────────
    # CRITICAL: lower() advances the per-function value-id counter `vf`,
    # so it must be called EXACTLY ONCE per insn. The 2026-05-02
    # GameMode-oscillation black-screen regression was a duplicate
    # lower() bug — the pre-pass and the emit loop both called lower(),
    # bumping vids out of sync with codegen's _v() mapping.
    #
    # block_per_insn_ir[key] holds (Insn, [IROp]) pairs — emit needs
    # the source Insn for Call dispatch detection.
    # block_ir[key] holds the flat list of IROps — used by the NLR
    # detector below.
    block_per_insn_ir: Dict[DecodeKey, List[Tuple[object, List[IROp]]]] = {}
    block_ir: Dict[DecodeKey, List[IROp]] = {}
    def _tail_call_stmt(call_expr: str, comment: str,
                        nlr_info_for_block: Optional[dict] = None,
                        *,
                        prefix: str = "",
                        tramp_pc24: Optional[int] = None) -> str:
        # Option-1 cpu->S ABI: a tail JMP/JML does NOT push a return frame;
        # the tail callee inherits THIS function's host-return validity (a
        # tail-call hands off our return obligation) and THIS function's
        # entry-S baseline (a split shared suffix may pop bytes pushed before
        # the tail transfer). NLR is no longer signalled via _pending_skip —
        # it flows through cpu->S + dispatch, so nlr_info_for_block is
        # ignored here.
        del nlr_info_for_block
        # Trampoline (same-bank tail-calls only, tramp_pc24 given): a DISPATCHED
        # frame (_hrv==0) running inside a cpu_dispatch_pc_from driving loop must
        # NOT make a nested direct C tail-call — a fall-through chain (ActRaiser
        # $8915->$8966 object loop) then nests the C/recomp stack once per object
        # and overflows RECOMP_STACK_DEPTH=64 (-> over-unwind SKIP, black
        # playfield). Instead inherit the entry-S baseline (so the callee's RTS
        # resolves against it) and yield to the driving loop via TAILCALL. A
        # PAIRED frame (_hrv==1) is the chain root: keep the direct call (it
        # drives, bounded depth). Cross-bank tail-calls keep the direct form.
        if tramp_pc24 is not None:
            T = f"0x{tramp_pc24 & 0xFFFFFF:06x}u"
            # _hrv==0: already inside a cpu_dispatch_pc_from driving loop ->
            #   inherit entry-S + yield via TAILCALL so it iterates flat.
            # _hrv==1: PAIRED (hardware-JSR) chain root with NO driving loop
            #   above. A direct nested C tail-call here re-nests the C/recomp
            #   stack once per iteration (ActRaiser $8915->$8966 object loop
            #   entered via $82E2's JSR $8915; hrv=1 inherited down the chain),
            #   leaking the SNES stack ~6B/frame -> underflow SIGSEGV. Instead
            #   establish a LOCAL driving loop here (cpu_dispatch_pc_from drives
            #   TAILCALL flat). The target runs dispatched (host_return_valid=0,
            #   Option-1) -- so do NOT set the inherit-context on this branch,
            #   or the target's prologue would re-adopt hrv=1 and re-nest.
            return (
                f"{prefix}{{ "
                f"if (!_hrv) {{ cpu->host_return_valid = _hrv; "
                f"cpu_tailcall_inherit_return_context(_entry_s, _hrv); "
                f"cpu_tailcall_request({T}, _entry_s, {T}); "
                f"RecompStackPop(); return RECOMP_RETURN_TAILCALL; }} "
                f"RecompStackPop(); "
                f"return cpu_dispatch_pc_from(cpu, {T}, _entry_s, {T}); }}  {comment}"
            )
        return (
            f"{prefix}{{ cpu->host_return_valid = _hrv; "
            f"cpu_tailcall_inherit_return_context(_entry_s, _hrv); "
            f"RecompReturn _tc = {call_expr}; "
            f"RecompStackPop(); return _tc; }}  {comment}"
        )

    for key in block_order:
        pairs: List[Tuple[object, List[IROp]]] = []
        flat: List[IROp] = []
        for di in cfg.blocks[key].insns:
            ops = lower(di.insn, value_factory=vf)
            pairs.append((di.insn, ops))
            flat.extend(ops)
        block_per_insn_ir[key] = pairs
        block_ir[key] = flat

    # ── Non-local-return idiom detection ────────────────────────────────
    # A basic block is an NLR-block if its IR has the shape
    #   [<setup ops>] + [PullReg(A) × N] + (Goto | Return)
    # where N is a multiple of the return-PC byte count (2 for RTS, 3
    # for RTL). Setup ops are anything OTHER than Push/Pull/Call (those
    # would interfere with stack accounting). Three sub-cases:
    #   (a) Block ends in its own Return — single-block PLA*N RTS,
    #       e.g. a leaf "SkipCaller" function. Skip = N / unit.
    #   (b) Block ends in a Goto whose successor block ends in Return
    #       (no further stack manipulation in successor) — multi-block
    #       BNE NLRBlock; NLRBlock: PLA*N + BRA tail; tail: work + RTS.
    #   (c) Block has setup work + PLA*N + JMP cross-fn. The JMP target
    #       is decoded INTO this function's CFG (v2 inlines cross-fn
    #       branch targets), and that target eventually RTSes. The
    #       setup ops are real game-state changes — they MUST be
    #       emitted; only the PLAs and the SKIP emit are special.
    #       Yoshi-block 2026-05-02: $00:F005 has this pattern with
    #       JMP $EE35 — Mario-on-Yoshi-vs-koopa-slope death root.
    # Returns a dict {skip:int, pla_start_ir_idx:int, pla_count:int}
    # so the emit loop can preserve setup ops, skip just the PLAs, and
    # set _pending_skip before the terminator.
    import os, sys as _sys
    _NLR_DEBUG = os.environ.get('SNESRECOMP_NLR_DEBUG') == '1'
    def _detect_nlr(key: DecodeKey):
        def _dbg(msg):
            if _NLR_DEBUG:
                print(f'[nlr_dbg] {func_name} {_label_for(key)}: {msg}',
                      file=_sys.stderr, flush=True)
        ops = block_ir.get(key, [])
        if not ops:
            _dbg('REJECT: no ops')
            return None
        # Three terminator shapes:
        #   - Block ends in Return — the RTS picks up _pending_skip.
        #   - Block ends in Goto — the goto's successor must be a
        #     Return-terminated block.
        #   - Block ends in something else (e.g. PullReg, Read, Write) —
        #     the block has NO terminator IR, just an implicit
        #     fall-through to its single CFG successor. Same NLR
        #     handling as the Goto case (chase successor).
        last_op = ops[-1]
        if isinstance(last_op, (Goto, Return)):
            terminator = last_op
            scan_end_excl = len(ops) - 1
        else:
            terminator = None
            scan_end_excl = len(ops)
        # Walk backward from scan_end_excl, counting PullReg(A)s.
        i = scan_end_excl - 1
        pla_end_excl = scan_end_excl
        while i >= 0 and isinstance(ops[i], PullReg) and ops[i].reg == Reg.A:
            i -= 1
        pla_start = i + 1
        pull_count = pla_end_excl - pla_start
        _dbg(f'pull_count={pull_count} pla_start={pla_start} '
             f'terminator={type(terminator).__name__ if terminator else "fall-through"}')
        if pull_count < 2:
            return None
        # Setup region (before PLAs) must not contain Call/IndirectGoto —
        # those would interfere with stack accounting. Goto/CondBranch in
        # the middle would mean the block isn't straight-line, which isn't
        # possible at this stage of v2 (each basic block has exactly one
        # terminator), but check defensively.
        #
        # Push/Pull ops in the setup region are TOLERATED if they balance
        # against function-wide PHA/PLA accounting — see check below. This
        # handles the F971 idiom where a function does:
        #   PHA / PHA / ... / PLA STA / PLA STA / PLA PLA RTS
        # The first two PLAs are paired with the two PHAs (intra-function
        # balance); the trailing PLA PLA is the NLR skip. The old per-block
        # "no push/pull in setup" check rejected this pattern, causing the
        # recomp to emit literal pops that consumed caller-frame bytes.
        # 2026-05-21 fix for Zelda camera axis-swap bug (see ISSUES.md).
        setup_pushes = 0
        setup_pulls = 0
        for op in ops[:pla_start]:
            if isinstance(op, (Call, IndirectGoto)):
                _dbg(f'REJECT: setup region has {type(op).__name__}')
                return None
            if isinstance(op, (Goto, Return, CondBranch)):
                _dbg(f'REJECT: setup region has {type(op).__name__}')
                return None
            if isinstance(op, (PushReg, Push)):
                setup_pushes += 1
            elif isinstance(op, (PullReg, Pull)):
                setup_pulls += 1
        # Determine return-PC byte count: 2 for RTS, 3 for RTL.
        long_return = None
        if isinstance(terminator, Return):
            long_return = terminator.long
        else:
            # Goto OR implicit fall-through: chase the lone successor.
            # Must be a local block ending in Return (no further stack
            # manipulation in successor — its RTS will pick up the
            # _pending_skip we set here).
            block = cfg.blocks[key]
            succs = block.successors
            if isinstance(terminator, Goto) and len(succs) == 0:
                # Cross-function tail jump after PLA*N. The common shape is:
                #
                #   ... setup ...
                #   PLA
                #   PLA
                #   JML target
                #
                # The PLAs discard this function's caller return frame, then
                # control transfers to another function. There is no local
                # successor Return for the original detector to chase, but the
                # effect is still a non-local return: after the tail target
                # returns, the C caller corresponding to the popped frame must
                # be skipped. Accept only explicit long JMP/JML terminators so
                # malformed local CFG gaps do not get treated as NLRs.
                last_insn = block.insns[-1].insn if block.insns else None
                if (getattr(last_insn, 'mnem', '') == 'JMP'
                        and getattr(last_insn, 'length', 0) == 4):
                    if pull_count % 2 == 0:
                        long_return = False
                    elif pull_count % 3 == 0:
                        long_return = True
                    else:
                        _dbg(f'REJECT: cross-tail pull_count={pull_count} '
                             'not divisible by 2 or 3')
                        return None
                    unit = 3 if long_return else 2
                    skip = pull_count // unit
                    if skip < 1 or skip > 3:
                        _dbg(f'REJECT: cross-tail skip={skip} out of [1,3]')
                        return None
                    _dbg(f'ACCEPT: cross-tail skip={skip} '
                         f'pla_start={pla_start} pla_count={pull_count}')
                    return {
                        'skip': skip,
                        'pla_start_ir_idx': pla_start,
                        'pla_count': pull_count,
                        'cross_tail': True,
                    }
                _dbg('REJECT: no local successor and not explicit long JMP')
                return None
            _dbg(f'no Return; successors={[_label_for(s) for s in succs]}')
            if len(succs) != 1:
                _dbg(f'REJECT: succ_count={len(succs)} (not exactly 1)')
                return None
            succ_key = succs[0]
            if succ_key not in block_ir:
                _dbg(f'REJECT: succ {_label_for(succ_key)} not in block_ir')
                return None
            succ_ops = block_ir[succ_key]
            if not succ_ops:
                _dbg(f'REJECT: succ has no ops')
                return None
            last = succ_ops[-1]
            if not isinstance(last, Return):
                _dbg(f'REJECT: succ last op = {type(last).__name__} (not Return)')
                return None
            # Successor must not contain Push/Pull (they'd interfere with
            # stack accounting). Call/IndirectGoto are tolerated — the
            # callee is balanced and doesn't affect THIS function's stack
            # frame relative to entry. Old detector pre-2026-05-02 only
            # rejected Push/Pull/Call but not IndirectGoto; my prior
            # tightening to also reject Call regressed the A3CB pattern
            # (its successor block has ALU/load/store ops only, no Call,
            # but the over-tightening in earlier draft erroneously also
            # required absence of Call which DOES match real NLR shapes
            # like the koopa-shell fix — keep loose like the original).
            for sop in succ_ops:
                if isinstance(sop, (PushReg, PullReg, Push, Pull)):
                    _dbg(f'REJECT: succ has {type(sop).__name__}')
                    return None
            long_return = last.long
        unit = 3 if long_return else 2
        if pull_count % unit != 0:
            _dbg(f'REJECT: pull_count={pull_count} not multiple of unit={unit}')
            return None
        skip = pull_count // unit
        if skip < 1 or skip > 3:
            _dbg(f'REJECT: skip={skip} out of [1,3]')
            return None
        # Function-wide PHA/PLA balance check (2026-05-21):
        # If the setup region of this block (or any other block, through
        # the trailing-PLA chase) contains PLAs that aren't paired with
        # PHAs in the same block, those PLAs MUST be paired with PHAs
        # elsewhere in the function. Otherwise the function's net stack
        # delta would consume MORE than just the NLR-skip bytes — the
        # `pull_count` we computed would not be the true "extra pops".
        #
        # Accept iff (function-wide PLAs) - (function-wide PHAs) == pull_count
        # (the only imbalance is the trailing NLR-skip pulls).
        if setup_pushes > 0 or setup_pulls > 0:
            fn_pushes = 0
            fn_pulls = 0
            for blk_ops in block_ir.values():
                for op in blk_ops:
                    if isinstance(op, (PushReg, Push)):
                        fn_pushes += 1
                    elif isinstance(op, (PullReg, Pull)):
                        fn_pulls += 1
            if fn_pulls - fn_pushes != pull_count:
                _dbg(f'REJECT: function-wide imbalance fn_pulls={fn_pulls} '
                     f'fn_pushes={fn_pushes} pull_count={pull_count}')
                return None
        _dbg(f'ACCEPT: skip={skip} pla_start={pla_start} pla_count={pull_count}')
        return {
            'skip': skip,
            'pla_start_ir_idx': pla_start,
            'pla_count': pull_count,
        }

    # ── Extended NLR detector: PLA*N at block START + branching tail ──
    #
    # Sub-case (d), 2026-05-14: PLA*N at the START of a block followed
    # by intermediate stateless logic and a conditional or
    # unconditional terminator whose every forward-reachable path
    # eventually reaches an RTS/RTL without any intervening
    # Push/Pull/Call/IndirectGoto. The PLAs eat the parent's return
    # address; the subsequent logic runs; the eventual RTS uses the
    # grandparent's return address.
    #
    # Canonical case: $00:9AEA HandleSelectionCursor.CheckMovement —
    #   PLA PLA              ; eat parent return
    #   LDA $15 / AND / LSR  ; cursor-direction logic
    #   BEQ .Return          ; branch to RTS
    #   ... cursor-move logic ...
    # .Return: RTS
    #
    # The existing detector (cases a/b/c) misses this because the
    # PLAs aren't immediately adjacent to a Return/Goto terminator —
    # they're at the start of the block, with stateless logic + a
    # CondBranch between them and the function's RTS.
    #
    # Detection is conservative: rejects if ANY reachable block has
    # a Push/Pull/Call/IndirectGoto, or if any path is dead-ended
    # (no terminator, no Return).
    def _detect_nlr_at_start(key: DecodeKey):
        def _dbg(msg):
            if _NLR_DEBUG:
                print(f'[nlr_dbg_d] {func_name} {_label_for(key)}: {msg}',
                      file=_sys.stderr, flush=True)
        ops = block_ir.get(key, [])
        if not ops:
            return None
        # Count PLA*N at the START.
        pla_end = 0
        while pla_end < len(ops) and isinstance(ops[pla_end], PullReg) \
                and ops[pla_end].reg == Reg.A:
            pla_end += 1
        pull_count = pla_end
        if pull_count < 2:
            return None
        # Determine terminator presence and trailing scan boundary.
        last_op = ops[-1]
        if isinstance(last_op, (Goto, Return, CondBranch)):
            scan_end_excl = len(ops) - 1
            terminator = last_op
        else:
            scan_end_excl = len(ops)
            terminator = None
        # Middle region (between PLAs and terminator): no stack ops, no
        # call, no nested goto/return/condbranch (each block has exactly
        # one terminator at end — defensive).
        for op in ops[pla_end:scan_end_excl]:
            if isinstance(op, (PushReg, PullReg, Push, Pull)):
                _dbg(f'REJECT: middle has {type(op).__name__}')
                return None
            if isinstance(op, (Call, IndirectGoto)):
                _dbg(f'REJECT: middle has {type(op).__name__}')
                return None
            if isinstance(op, (Goto, Return, CondBranch)):
                _dbg(f'REJECT: middle has {type(op).__name__}')
                return None
        # If the block ends in its own Return, skip = pull_count / unit
        # and we're done — but that's case (a) which the primary
        # detector already caught. Bail to avoid double-detection.
        if isinstance(terminator, Return):
            return None
        # All forward-reachable paths from this block's successors
        # must end in Return with no Push/Pull/Call along the way.
        # DFS the successor graph; memoise.
        unit_seen: List[int] = []  # collected return widths

        def _forward_clean(start_key: DecodeKey) -> bool:
            stack = [start_key]
            visited: set = set()
            while stack:
                k = stack.pop()
                if k in visited:
                    continue
                visited.add(k)
                if k not in block_ir:
                    _dbg(f'REJECT: succ {_label_for(k)} not in block_ir')
                    return False
                k_ops = block_ir[k]
                if not k_ops:
                    _dbg(f'REJECT: succ {_label_for(k)} empty')
                    return False
                k_last = k_ops[-1]
                # Scan ops up to but not including the terminator.
                if isinstance(k_last, (Goto, Return, CondBranch)):
                    scan_n = len(k_ops) - 1
                else:
                    scan_n = len(k_ops)
                for op in k_ops[:scan_n]:
                    if isinstance(op, (PushReg, PullReg, Push, Pull,
                                       Call, IndirectGoto)):
                        _dbg(f'REJECT: succ {_label_for(k)} has '
                             f'{type(op).__name__}')
                        return False
                if isinstance(k_last, Return):
                    unit_seen.append(3 if k_last.long else 2)
                    continue
                # Goto / CondBranch / fall-through: walk successors.
                k_succs = cfg.blocks[k].successors
                if not k_succs:
                    _dbg(f'REJECT: succ {_label_for(k)} no successors '
                         f'and no Return')
                    return False
                for s in k_succs:
                    stack.append(s)
            return True

        block = cfg.blocks[key]
        succs = block.successors
        if not succs:
            _dbg('REJECT: no successors and no own Return')
            return None
        for s in succs:
            if not _forward_clean(s):
                return None
        if not unit_seen:
            _dbg('REJECT: no Return reached')
            return None
        # All reached Returns must use the same width (mixing RTS and
        # RTL in one function's NLR pattern would be exotic — bail).
        if len(set(unit_seen)) != 1:
            _dbg(f'REJECT: mixed return widths {unit_seen}')
            return None
        unit = unit_seen[0]
        if pull_count % unit != 0:
            _dbg(f'REJECT: pull_count={pull_count} not multiple of '
                 f'unit={unit}')
            return None
        skip = pull_count // unit
        if skip < 1 or skip > 3:
            _dbg(f'REJECT: skip={skip} out of [1,3]')
            return None
        _dbg(f'ACCEPT (case d): skip={skip} pla_count={pull_count}')
        return {
            'skip': skip,
            'pla_start_ir_idx': 0,
            'pla_count': pull_count,
            # case-d marker: PLAs are at the START of the block, not
            # the end. The emit code uses this to pick the right insn
            # indices when the terminator is a CondBranch (not handled
            # by the case-a/b/c "has_terminator_ir" path).
            'pla_at_start': True,
        }

    # Option-1 cpu->S return-frame ABI (see IMPROVEMENTS.md): the PLA*N NLR
    # idiom is now handled by emitting the PLAs as NORMAL cpu->S pops and
    # letting RTS/RTL consume the exposed return frame (host_return_valid +
    # cpu_dispatch_pc). The detector is retained for DIAGNOSTICS ONLY and no
    # longer drives emission, so `nlr_skip_by_block` stays EMPTY — the emit
    # loop then emits every PLA/PLP and every Return normally.
    nlr_skip_by_block: Dict[DecodeKey, dict] = {}
    nlr_diag_by_block: Dict[DecodeKey, dict] = {}  # diagnostics only; NOT used for emit
    for key in block_order:
        info = _detect_nlr(key)
        if info is None:
            info = _detect_nlr_at_start(key)
        if info is not None:
            nlr_diag_by_block[key] = info

    # Bank where THIS function's body lives. Used to compute the 24-bit
    # address of cross-function targets (which always lie within the same
    # bank — cross-BANK jumps go through Call/JSL machinery, not Goto).
    _SAME_BANK = bank & 0xFF

    def _goto_or_return(target: DecodeKey, prefix: str = "",
                         source_pc24: Optional[int] = None) -> str:
        label = _label_for(target)
        if label in local_labels:
            return f"{prefix}goto {label};"
        # HLE-replacement check: cfg `exclude_range S E` carves out a
        # data region — by convention, asm whose lifted form is replaced
        # by host-side HLE (e.g. the SMW asm main loop at $00:806B-$8078,
        # which the runner replaces with SmwRunOneFrameOfGame).
        # A cross-fn jump TARGETING such a range can never reach asm
        # code: there isn't any. The runner owns control-flow at that
        # address. The right shape is `return to host` — let the
        # outer-frame's HLE mechanism take over.
        # Skip auto-promote recording so the target doesn't get
        # synthesized as an empty BankEntry next pass.
        if exclude_ranges:
            tpc = target.pc & 0xFFFF
            for (lo, hi) in exclude_ranges:
                if lo <= tpc < hi:
                    return (
                        f"{prefix}return RECOMP_RETURN_NORMAL; "
                        f"/* {label} HLE-replaced "
                        f"(cfg exclude_range {lo:04X}-{hi:04X}) */"
                    )
        # cfg `tail_call:<addr>` directive — declared sibling fall-through.
        # When the decoder's `end:` boundary cuts a routine that
        # deliberately falls into a separately-named adjacent fn (real
        # ROM idiom: two callable entry points sharing a body), cfg
        # encodes the fact via `tail_call:`. The boundary edge becomes
        # an explicit tail call to the sibling fn instead of an
        # unresolvable goto. (m, x) come from the boundary DecodeKey
        # so the right variant suffix is used.
        if (tail_call_pc16 is not None
                and tail_call_target_name is not None
                and (target.pc & 0xFFFF) == (tail_call_pc16 & 0xFFFF)):
            sib_suffix = _variant_suffix(target.m, target.x)
            # Register cross-variant Call demand so v2_regen's auto-
            # promote synthesizes the (m, x) body when it differs from
            # the sibling's cfg-declared default. Without this, the
            # C linker fails with "unresolved external _M{m}X{x}" when
            # the boundary's (m, x) doesn't match the cfg-default of
            # the sibling — exactly what tripped the zelda3 cross-
            # variant externals (NMI_RunTileMapUpdateDMA_M0X1,
            # SpotlightInternal_M1X0, etc.) on the 2026-05-17
            # tail-call-past-end class fix.
            from v2.codegen import register_call_demand
            tail_pc24 = (_SAME_BANK << 16) | (target.pc & 0xFFFF)
            register_call_demand(tail_pc24, target.m, target.x)
            return _tail_call_stmt(
                f"{tail_call_target_name}{sib_suffix}(cpu)",
                f"/* tail_call into sibling fn at ${target.pc & 0xFFFF:04X} "
                f"(cfg tail_call: directive) */",
                prefix=prefix,
                tramp_pc24=tail_pc24,
            )

        # Tail-call past `end:` boundary into a declared sibling
        # function. 2026-05-17 class fix for the inline-cross-fn-blocks
        # gap: zelda3 (and SMW's adjacent-entrypoint idiom) has functions
        # that fall through directly into the next sibling function with
        # no RTS — the asm is balanced end-to-end, but the C boundary the
        # cfg/ingester drew has to be honoured. Without this case the
        # decoder would either (a) inline the whole sibling body and any
        # transitive fall-throughs into the current C function (the bug
        # behind the Zelda intro-loop submodule reset 2026-05-17), or
        # (b) emit cpu_trace_unresolved_goto_trap.
        #
        # Detection: the goto target's 24-bit PC resolves to a known
        # function entry in _NAME_RESOLVER (populated by v2_regen from
        # every `func` + `name` declaration across all bank cfgs). Same-
        # bank emits a direct call; cross-bank would also work but
        # cross-bank gotos don't reach this code path (they're emitted
        # via Call/JML machinery upstream).
        #
        # Variant: tail-call to the (m, x) of the target DecodeKey — the
        # decode state at the boundary. register_call_demand records the
        # demand so v2_regen's auto-promote synthesizes the variant if
        # it isn't already in cfg.
        target_pc24 = (_SAME_BANK << 16) | (target.pc & 0xFFFF)
        src_pc24 = source_pc24 if source_pc24 is not None else 0
        from v2.codegen import (get_name_for_pc, register_call_demand,
                                 valid_variant_list)
        sibling_name = get_name_for_pc(target_pc24)
        if sibling_name is not None:
            sib_suffix = _variant_suffix(target.m, target.x)
            register_call_demand(target_pc24, target.m, target.x)
            return _tail_call_stmt(
                f"{sibling_name}{sib_suffix}(cpu)",
                f"/* tail-call past end: into {sibling_name}{sib_suffix} "
                f"at ${target.pc & 0xFFFF:04X} */",
                prefix=prefix,
                tramp_pc24=target_pc24,
            )

        # Unresolvable cross-function jump.
        #
        # With the inline-cross-fn-blocks model (2026-05-02), the decoder
        # imports BRA/BRL/JMP-ABS/cond-branch targets that lie past the
        # cfg `end:` boundary directly into THIS function's CFG, so the
        # vast majority of intra-bank cross-fn jumps resolve as local
        # labels above. Anything that reaches HERE is one of:
        #   - a cross-BANK jump (JML/long-JMP — the decoder doesn't decode
        #     other banks; out of scope for in-bank inlining),
        #   - a target outside [0x8000, 0xFFFF] (data/header region),
        #   - a pathological cfg setup we couldn't import.
        #
        # We do NOT auto-promote the target into a separate C function:
        # that was the prior policy and it stranded PHB/PLB pairs across
        # C scopes (caused DB=$C0 at dispatch entry — the title-screen
        # regression). Auto-promote only synthesizes for true subroutine
        # entries (JSR/JSL targets), not for arbitrary jump destinations.
        #
        # 2026-05-03 Step 2-A: emit no longer silently returns
        # RECOMP_RETURN_NORMAL. Goes through cpu_trace_unresolved_goto_trap
        # which captures (gen function name + source_pc24 + target label)
        # so a hit unambiguously identifies WHICH gen variant ran the
        # unresolvable goto — disambiguating sibling variants that
        # happen to share source PCs. The trap returns NORMAL after
        # capture (Release path) or aborts (Oracle/debug); see
        # runner/src/cpu_trace.c.
        return (
            f"{prefix}return cpu_trace_unresolved_goto_trap(cpu, "
            f"0x{src_pc24:06X}, 0x{target_pc24:06X}, "
            f"\"{func_name}\", \"{label}\");"
            f" /* {label} unresolvable cross-fn goto — "
            f"target outside this bank's import range */"
        )

    for key in block_order:
        block = cfg.blocks[key]
        lines: List[str] = []
        block_terminated = False  # True if last op was branch/goto/return/call

        # Iterate the pre-lowered (Insn, [IROp]) pairs. Calling lower()
        # again here would mint fresh Value-ids and break codegen's
        # vid → C-var mapping (GameMode00↔01 oscillation 2026-05-02).
        #
        # Pre-scan #1: identify JML-with-dispatch_entries (PHK+PER+JML or
        # PHK+PEA+JML inline-dispatch trampoline; e.g. GenerateTile_Dispatch
        # at $00:BFBC). The PHK / PEA / PER immediately preceding such a JML
        # are TRAMPOLINE SETUP that the runtime dispatcher (ExecutePtr /
        # ExecutePtrLong) would consume — under the synthesized C switch
        # we inline the dispatch directly, so those pushes would leak 3
        # garbage bytes onto the simulated SNES stack. Skip emit for them.
        # Yoshi-block freeze ROOT 2026-05-02 — see project memory.
        #
        # Pre-scan #2: NLR PLA idiom. If this block was detected as NLR
        # (PLA*N at the END of the block before a Goto/Return), figure
        # out the INSN-LEVEL indices of the PLA insns so we can skip
        # their literal-pop emit. The setup ops BEFORE the PLAs MUST
        # still emit — they're real game-state changes (e.g. STA $1DFC
        # in $00:F005's L_F024 block sets the Yoshi-knockoff sound).
        # The SKIP_N _pending_skip set is injected before the terminator
        # insn so the eventual RTS picks it up.
        pairs = block_per_insn_ir.get(key, [])
        skip_emit_idx = set()
        for ji, (jdi, _jops) in enumerate(pairs):
            if (jdi.mnem == 'JMP' and jdi.length == 4
                    and getattr(jdi, 'dispatch_entries', None)):
                k2 = ji - 1
                while k2 >= 0:
                    prev = pairs[k2][0]
                    if prev.mnem in ('PHK', 'PEA', 'PER'):
                        skip_emit_idx.add(k2)
                        k2 -= 1
                    else:
                        break

        # NLR pre-scan: map the IR-level pla_start to insn-level indices.
        # Each PLA insn lowers to exactly one PullReg(A) IR op. The
        # PLA insns are the LAST `pla_count` insns of the IR's PLA run.
        # Three terminator shapes (matching detector):
        #   - Block ends in Return-IR (RTS/RTL): last insn is RTS/RTL,
        #     PLAs are insns [last - pla_count, last).
        #   - Block ends in Goto-IR (JMP/BRA): same as above.
        #   - Block ends WITHOUT terminator IR (implicit fall-through):
        #     last insn IS one of the PLAs; PLAs are insns [last - pla_count + 1, last + 1).
        # We use the IR's `terminator` flag (None vs Goto/Return) plus
        # mnemonic spot-check.
        nlr_info = nlr_skip_by_block.get(key)
        nlr_pla_insn_indices = set()
        nlr_inject_before_idx = None  # insn index BEFORE which to inject SKIP setter
        nlr_inject_after_loop = False  # if True, emit SKIP after the per-insn loop
        if nlr_info is not None:
            pla_count = nlr_info['pla_count']
            ir_ops_flat = block_ir.get(key, [])
            if nlr_info.get('pla_at_start'):
                # Case-d: PLAs are the FIRST `pla_count` insns of the
                # block; terminator can be Goto, Return, or CondBranch
                # (any). Inject the SKIP setter BEFORE the terminator
                # insn so its eventual RTS picks it up — works for
                # CondBranch too since both legs reach a clean Return.
                pla_first_insn = 0
                pla_last_excl = pla_count
                last_ir = ir_ops_flat[-1] if ir_ops_flat else None
                has_terminator_ir = isinstance(
                    last_ir, (Goto, Return, CondBranch))
                if has_terminator_ir:
                    inject_at = len(pairs) - 1
                else:
                    inject_at = None
            else:
                has_terminator_ir = (
                    len(ir_ops_flat) > 0
                    and isinstance(ir_ops_flat[-1], (Goto, Return))
                )
                if has_terminator_ir:
                    term_insn_idx = len(pairs) - 1
                    pla_first_insn = term_insn_idx - pla_count
                    pla_last_excl = term_insn_idx
                    inject_at = term_insn_idx  # before the terminator insn
                else:
                    pla_first_insn = len(pairs) - pla_count
                    pla_last_excl = len(pairs)
                    inject_at = None  # no terminator insn → emit after loop
            if pla_first_insn < 0:
                nlr_info = None
            else:
                ok = True
                for ix in range(pla_first_insn, pla_last_excl):
                    if pairs[ix][0].mnem != 'PLA':
                        if _NLR_DEBUG:
                            print(f'[nlr_dbg] {func_name} {_label_for(key)}: '
                                  f'mnem-check FAIL at insn {ix}: '
                                  f'{pairs[ix][0].mnem!r} (expected PLA); '
                                  f'pla_first={pla_first_insn} pla_last_excl={pla_last_excl} '
                                  f'has_term={has_terminator_ir} pla_count={pla_count}',
                                  file=_sys.stderr, flush=True)
                        ok = False
                        break
                if not ok:
                    nlr_info = None
                else:
                    for ix in range(pla_first_insn, pla_last_excl):
                        nlr_pla_insn_indices.add(ix)
                        skip_emit_idx.add(ix)
                    nlr_inject_before_idx = inject_at
                    nlr_inject_after_loop = (inject_at is None)

        if _NLR_DEBUG and key in nlr_skip_by_block:
            print(f'[nlr_dbg] EMIT-LOOP {func_name} {_label_for(key)}: '
                  f'nlr_info={nlr_info!r} pla_indices={sorted(nlr_pla_insn_indices)} '
                  f'inject_before={nlr_inject_before_idx} after_loop={nlr_inject_after_loop} '
                  f'skip_emit_idx={sorted(skip_emit_idx)}',
                  file=_sys.stderr, flush=True)
        for ii, (di_insn, ir_ops) in enumerate(pairs):
            # Call-site context (AR_CALLMX, 2026-06-30): tell codegen the
            # current instruction's own 24-bit address so _emit_call can
            # attribute a mid-function m/x invariant check to this exact
            # site. See codegen.set_current_site_pc24.
            from v2.codegen import set_current_site_pc24
            set_current_site_pc24(di_insn.addr & 0xFFFFFF)
            # NLR: inject _pending_skip setter + diagnostics RIGHT BEFORE
            # the terminator insn. This ensures any preceding setup ops
            # have already emitted, and the upcoming Goto/Return picks up
            # the SKIP value.
            if nlr_info is not None and ii == nlr_inject_before_idx:
                skip = nlr_info['skip']
                block_pc24 = (bank << 16) | (key.pc & 0xFFFF)
                site_label = f"{func_name}/{_label_for(key)}"
                lines.append(
                    f"cpu_trace_nlr_site_exec(cpu, 0x{block_pc24:06X}, "
                    f"\"{site_label}\");"
                )
                lines.append(
                    f"cpu_trace_event(cpu, 0, CPU_TR_NLR_DETECT, "
                    f"(uint8){skip}, 0); /* PLA*N + (Goto|RTS) = "
                    f"return-to-grandparent via SKIP_{skip} */"
                )
                lines.append(f"_pending_skip = RECOMP_RETURN_SKIP_{skip};")
                lines.append(
                    f"cpu_trace_pending_skip_write(cpu, 0x{block_pc24:06X}, "
                    f"(uint8)RECOMP_RETURN_SKIP_{skip}, \"{func_name}\");"
                )

            if ii in skip_emit_idx:
                if ii in nlr_pla_insn_indices:
                    lines.append(
                        f"/* PLA skipped — NLR idiom; SKIP_{nlr_info['skip']} "
                        f"set on _pending_skip above */"
                    )
                else:
                    lines.append(
                        f"/* trampoline setup {di_insn.mnem} skipped — "
                        f"inlined into synthesized dispatch below */"
                    )
                continue
            for op in ir_ops:
                if isinstance(op, CondBranch):
                    # Cond branch: block has TWO successors: fall-through (0)
                    # and taken-target (1) per _successors() ordering.
                    succs = block.successors
                    fall = succs[0] if len(succs) >= 1 else None
                    taken = succs[1] if len(succs) >= 2 else None
                    pred = f"{_reg_for_flag(op.flag)} == {op.take_if}"
                    blk_pc24 = (bank << 16) | (key.pc & 0xFFFF)
                    if taken is not None:
                        target_stmt = _goto_or_return(taken, source_pc24=blk_pc24)
                        lines.append(f"if ({pred}) {{ {target_stmt} }}")
                    if fall is not None:
                        lines.append(_goto_or_return(fall, source_pc24=blk_pc24)
                                     + " /* fall-through */")
                        block_terminated = True
                elif isinstance(op, Goto):
                    # JML to a registered dispatch helper (ExecutePtr /
                    # ExecutePtrLong). Bytes after the JML are a function-
                    # pointer table; the decoder already read them into
                    # insn.dispatch_entries. Synthesize a C switch that
                    # directly invokes each handler — this replaces the
                    # asm trampoline and keeps the simulated SNES stack
                    # balanced (PHK+PER setup was skipped above).
                    insn = di_insn
                    if getattr(insn, 'dispatch_entries', None):
                        from v2.codegen import _emit_dispatch
                        for ln in _emit_dispatch(insn):
                            lines.append(ln)
                        block_terminated = True
                    else:
                        succs = block.successors
                        blk_pc24 = (bank << 16) | (key.pc & 0xFFFF)
                        if len(succs) >= 1:
                            lines.append(_goto_or_return(succs[0],
                                                          source_pc24=blk_pc24))
                        else:
                            # Cross-bank JML / out-of-import-range goto.
                            # 2026-05-18 class fix: if the JML operand
                            # resolves to a registered named function in
                            # ANY bank, emit a tail-call to its
                            # M{m}X{x} variant (using the current block's
                            # (m, x) — JML doesn't touch the status
                            # register's M/X bits). This is the cross-
                            # bank analogue of the same-bank tail-call-
                            # past-end: case handled in _goto_or_return.
                            # Closes the zelda Intro_FadeInBg →
                            # Palette_FadeIntro2 JML $00:ED8F gap that
                            # was producing a `return RECOMP_RETURN_NORMAL;
                            # /* Goto with no successor */` stub (block
                            # silently bailed instead of decrementing
                            # palette_filter_countdown — title screen
                            # never advanced to attract demo).
                            target_pc24 = None
                            if (getattr(insn, 'mnem', '') == 'JMP'
                                    and getattr(insn, 'mode', None) is not None):
                                from v2.lowering import LONG as _LONG_MODE
                                if insn.mode == _LONG_MODE:
                                    target_pc24 = insn.operand & 0xFFFFFF
                            if target_pc24 is not None:
                                from v2.codegen import (get_name_for_pc,
                                                          register_call_demand)
                                tgt_name = get_name_for_pc(target_pc24)
                                if tgt_name is not None:
                                    sib_suffix = _variant_suffix(key.m, key.x)
                                    register_call_demand(target_pc24,
                                                          key.m, key.x)
                                    tgt_bank = (target_pc24 >> 16) & 0xFF
                                    # JML changes PB on real hardware
                                    # (the destination bank byte becomes
                                    # the new PB). A same-bank tail-call
                                    # past end: doesn't need this — BRA
                                    # / BRL / JMP-abs leave PB
                                    # unchanged. For cross-bank we MUST
                                    # set cpu->PB so the callee's PB-
                                    # relative addressing (PC-rel
                                    # branches, K-byte reads) computes
                                    # against the right bank. We do NOT
                                    # save/restore: this is a tail-call,
                                    # the callee's RTL returns out to
                                    # the caller's caller (just like the
                                    # asm did) — that caller's JSL
                                    # wrapper restores PB itself.
                                    lines.append(
                                        f"cpu->PB = 0x{tgt_bank:02X}; /* JML "
                                        f"into bank ${tgt_bank:02X} */"
                                    )
                                    lines.append(_tail_call_stmt(
                                        f"{tgt_name}{sib_suffix}(cpu)",
                                        f"/* tail-call cross-bank into "
                                        f"{tgt_name}{sib_suffix} at "
                                        f"${target_pc24:06X} (JML "
                                        f"unresolved successor) */",
                                        nlr_info,
                                    ))
                                    block_terminated = True
                                    break  # exit `for op in ir_ops`
                            # No name resolved yet — for a CROSS-BANK
                            # JML the target is unambiguously a tail-call
                            # destination (PB changes; current function's
                            # successor block is past `end:` or in another
                            # bank). Register it as a Call demand so the
                            # next auto-promote pass synthesizes a func
                            # entry at the target. Then RE-EMIT this site
                            # as a placeholder tail-call to the synthesized
                            # `bank_BB_AAAA_MmXx` name — v2_regen will
                            # have created the entry by the next pass.
                            #
                            # This is the cross-bank tail-call class fix:
                            # we don't want to fall through to a trap and
                            # block the build on first emit; we want the
                            # demand pipeline to close the gap.
                            # In-function (same-bank) Gotos are NOT
                            # promoted (see the v2_regen.py "Goto targets
                            # are no longer auto-promoted" comment) — that
                            # would split asm routines. CROSS-bank is the
                            # safe subset: the bank change implies a
                            # function boundary.
                            src_pc24 = blk_pc24
                            tgt_str = (f"0x{target_pc24:06X}"
                                       if target_pc24 is not None else "0x000000")
                            if (target_pc24 is not None
                                    and ((target_pc24 >> 16) & 0xFF) != bank):
                                from v2.codegen import (
                                    register_call_demand,
                                    _is_invalid_lorom_call_target,
                                    get_name_for_pc)
                                if (_is_invalid_lorom_call_target(target_pc24)
                                        and get_name_for_pc(target_pc24)
                                        is None):
                                    # Cross-bank JML to a non-code address
                                    # (beyond ROM extent or < $8000 RAM):
                                    # the decoder followed a garbage operand
                                    # past an RTS — data decoded as code.
                                    # Don't auto-promote a function at a
                                    # non-code target; emit a no-op return,
                                    # mirroring _emit_call's skip for this
                                    # exact class. (Would crash on real HW
                                    # too — never reached in correct play.)
                                    lines.append(
                                        f"return RECOMP_RETURN_NORMAL; "
                                        f"/* cross-bank JML to "
                                        f"${target_pc24:06X} skipped — not a "
                                        f"valid LoROM code address (decoder "
                                        f"followed garbage operand past an "
                                        f"RTS) */")
                                else:
                                    register_call_demand(target_pc24,
                                                          key.m, key.x)
                                    sib_suffix = _variant_suffix(key.m, key.x)
                                    tgt_bank = (target_pc24 >> 16) & 0xFF
                                    tgt16 = target_pc24 & 0xFFFF
                                    synth_name = (
                                        f"bank_{tgt_bank:02X}_{tgt16:04X}"
                                    )
                                    lines.append(
                                        f"cpu->PB = 0x{tgt_bank:02X}; /* JML "
                                        f"into bank ${tgt_bank:02X} */"
                                    )
                                    lines.append(_tail_call_stmt(
                                        f"{synth_name}{sib_suffix}(cpu)",
                                        f"/* tail-call cross-bank into "
                                        f"{synth_name}{sib_suffix} at "
                                        f"${target_pc24:06X} (auto-promoted "
                                        f"via Call demand) */",
                                        nlr_info,
                                    ))
                            else:
                                lines.append(
                                    f"return cpu_trace_unresolved_goto_trap(cpu, "
                                    f"0x{src_pc24:06X}, {tgt_str}, "
                                    f"\"{func_name}\", \"L_{key.pc & 0xFFFF:04X}_"
                                    f"M{key.m}X{key.x}\");"
                                    f"  /* unresolvable cross-bank goto — "
                                    f"no named function at target */"
                                )
                        block_terminated = True
                elif isinstance(op, Return):
                    if (getattr(di_insn, 'dispatch_kind', None) == 'rts_trick'
                            and getattr(di_insn, 'dispatch_entries', None)):
                        # rts_dispatch (RTS-trick, e.g. $03:9156's relocated-
                        # stack chain): this RTS pops a runtime-pushed
                        # CONTINUATION address and jumps to it+1, which the
                        # decoder decoded as in-function blocks. PEEK the
                        # popped 16-bit return value; for a known target, pop
                        # the frame (S+=2) and `goto` the in-function label —
                        # so cpu->S and the whole chain's relocated-stack state
                        # are preserved (the fix the separate-`func` slicing
                        # couldn't give). Anything else falls through to the
                        # normal return below (the chain's TRUE terminal exit
                        # to the original caller is a separate, normal RTS in
                        # the dispatched continuation's own decoded path).
                        lines.append("{ uint16 _rts_s = cpu->S;")
                        lines.append(
                            "  uint16 _rts_t = (uint16)(((cpu_read8(cpu, 0x00, "
                            "(uint16)(_rts_s + 2)) << 8) | cpu_read8(cpu, 0x00, "
                            "(uint16)(_rts_s + 1))) + 1);")
                        lines.append("  switch (_rts_t) {")
                        # Label suffix must use the RTS INSN's decoded (m,x),
                        # NOT the block key's: a mid-block SEP/REP between the
                        # block leader and the RTS (e.g. $03:9EEE SEP #$20 with
                        # the block starting at $9EDC, m=0) means the decoder
                        # added + decoded the dispatch targets at the insn's
                        # flags (L_xxxx_M1X0) while the block key still says
                        # M0X0 — the old key-based lookup then found no label
                        # and the guard silently emitted an empty switch (every
                        # dispatch fell to default = the exact silent-skip the
                        # directive exists to prevent).
                        _im = di_insn.m_flag & 1
                        _ix = di_insn.x_flag & 1
                        for _entry in di_insn.dispatch_entries:
                            _t16 = _entry & 0xFFFF
                            _tgt = f"L_{_t16:04X}_M{_im}X{_ix}"
                            # Only emit a case whose target block was actually
                            # decoded IN THIS VARIANT. Wrong-width sibling
                            # variants (e.g. the M1Xx garbage decode of an
                            # m=0-only region) can reach this RTS site without
                            # ever decoding the continuation block; an
                            # unconditional `goto` then references an
                            # undeclared label -> build failure (seen 2026-07-04
                            # with `rts_dispatch 93AA 9316`: L_9316_M1X0 emitted
                            # in bank_03 M1X0/M1X1 variants that never decode
                            # $9316). Falling through to the default (normal
                            # return) is the correct behavior for a variant
                            # that can't host the continuation.
                            if _tgt not in local_labels:
                                lines.append(
                                    f"    /* case 0x{_t16:04X}: {_tgt} not "
                                    f"decoded in this variant -> default */")
                                continue
                            lines.append(
                                f"    case 0x{_t16:04X}: "
                                f"cpu->S = (uint16)(_rts_s + 2); goto {_tgt};")
                        # Unregistered continuation target: the chain pushed a
                        # CONTINUATION addr we never listed in `rts_dispatch`, so
                        # we cannot `goto` an in-function label and instead fall
                        # through to the generic host-return below — which, after
                        # this RTS's SEP/REP, can leak the wrong m/x to the caller
                        # (the $03:9156 act->sim M0X0 leak class). Log it (env-
                        # gated, default silent) so a missing target is named at
                        # runtime: add it to the `rts_dispatch` directive.
                        _site24 = di_insn.addr & 0xFFFFFF
                        lines.append(
                            "    default:")
                        lines.append(
                            "      if (getenv(\"AR_RTSDISP_MISS\"))")
                        lines.append(
                            "        fprintf(stderr, \"[rts_dispatch_miss] site=$"
                            f"{_site24:06X} popped target=$%04X "
                            "(UNREGISTERED -> generic return; add to "
                            "rts_dispatch) S=$%04X m=%d x=%d\\n\", "
                            "(unsigned)_rts_t, (unsigned)cpu->S, "
                            "(int)cpu->m_flag, (int)cpu->x_flag);")
                        lines.append(
                            "      break;  /* unknown -> normal return */")
                        lines.append("  } }")
                    for ln in emit_op(op):
                        lines.append(ln)
                    block_terminated = True
                elif isinstance(op, IndirectGoto):
                    # cfg-resolved (or auto-recovered) IndirectGoto: the
                    # decoder stamped `dispatch_entries` + `dispatch_idx_reg`
                    # on the insn. Route through _emit_indirect_dispatch
                    # to emit a real switch with direct tail-calls. Class
                    # fix for the IndirectGoto stub class — never emit a
                    # `/* IndirectGoto */` stub here. Unresolved sites
                    # are caught at v2_regen via graph.unresolved_indirects.
                    insn = di_insn
                    if getattr(insn, 'dispatch_entries', None):
                        from v2.codegen import _emit_indirect_dispatch
                        for ln in _emit_indirect_dispatch(insn):
                            lines.append(ln)
                        block_terminated = True
                    else:
                        # No resolution. Two sub-cases:
                        #   (a) cfg `hle_dispatch <pc16> <c_helper>` claims
                        #       this site — emit a tail-call to the named
                        #       C helper (host-side dispatcher).
                        #   (b) Otherwise emit the runtime trap so the
                        #       site is captured if execution ever reaches
                        #       it. NOT a stub (no silent fall-through);
                        #       requires HLE follow-up.
                        site_pc24 = (insn.addr & 0xFFFFFF) if insn is not None else \
                                    ((bank << 16) | (key.pc & 0xFFFF))
                        site_pc16 = site_pc24 & 0xFFFF
                        if hle_dispatch and site_pc16 in hle_dispatch:
                            c_helper = hle_dispatch[site_pc16]
                            lines.append(
                                f"{{ extern RecompReturn {c_helper}(CpuState *cpu); "
                                f"RecompReturn _r = {c_helper}(cpu); "
                                f"RecompStackPop(); return _r; }} "
                                f"/* hle_dispatch ${site_pc24:06X} — "
                                f"host-side dispatcher */"
                            )
                        else:
                            lines.append(
                                f"return cpu_trace_dispatch_oob(cpu, "
                                f"0x{site_pc24:06x}, 0xFFFF); "
                                f"/* unresolved IndirectGoto — HLE pending */")
                        block_terminated = True
                elif isinstance(op, PushReg) and getattr(
                        di_insn, 'dispatch_entries', None):
                    # RTS-stack dispatch: the decoder marked the PHA
                    # that would normally push target-1 for a following
                    # RTS. Emit a switch instead of the literal stack
                    # push, otherwise the synthesized return address
                    # leaks onto the simulated SNES stack.
                    from v2.codegen import _emit_indirect_dispatch
                    for ln in _emit_indirect_dispatch(di_insn):
                        lines.append(ln)
                    block_terminated = True
                elif isinstance(op, Call):
                    # Dispatch-helper JSL: the decoder marked the insn
                    # with `dispatch_entries`. Two routes:
                    #   (a) JSL ExecutePtr helper — index by A, switch
                    #       through `_emit_dispatch`.
                    #   (b) JSR (abs,X) authorised by `indirect_dispatch`
                    #       directive — index by X (or Y if cfg said so),
                    #       switch through `_emit_indirect_dispatch`.
                    # JSL ExecutePtr helpers are terminators. JSR
                    # (abs,X) dispatches are ordinary calls: the selected
                    # handler RTSes back to the next instruction.
                    insn = di_insn
                    if getattr(insn, 'dispatch_entries', None):
                        if getattr(insn, 'dispatch_idx_reg', None) in ('X', 'Y'):
                            from v2.codegen import _emit_indirect_dispatch
                            for ln in _emit_indirect_dispatch(insn):
                                lines.append(ln)
                            if getattr(insn, 'mnem', '') != 'JSR':
                                block_terminated = True
                        else:
                            from v2.codegen import _emit_dispatch
                            for ln in _emit_dispatch(insn):
                                lines.append(ln)
                            block_terminated = True
                    else:
                        for ln in emit_op(op):
                            lines.append(ln)
                else:
                    # ReadReg, ALU, Read/Write, etc. — non-terminating.
                    for ln in emit_op(op):
                        lines.append(ln)
        # NLR with no terminator IR (block IR was pure-PullReg, like
        # $01:A3CB's [PLA, PLA, fall-through]). The SKIP setter wasn't
        # injected during the per-insn loop because there was no
        # terminator insn to anchor on. Inject it here, BEFORE the
        # implicit fall-through emission below.
        if nlr_info is not None and nlr_inject_after_loop and not block_terminated:
            skip = nlr_info['skip']
            block_pc24 = (bank << 16) | (key.pc & 0xFFFF)
            site_label = f"{func_name}/{_label_for(key)}"
            lines.append(
                f"cpu_trace_nlr_site_exec(cpu, 0x{block_pc24:06X}, "
                f"\"{site_label}\");"
            )
            lines.append(
                f"cpu_trace_event(cpu, 0, CPU_TR_NLR_DETECT, "
                f"(uint8){skip}, 0); /* PLA*N + fall-through = "
                f"return-to-grandparent via SKIP_{skip} */"
            )
            lines.append(f"_pending_skip = RECOMP_RETURN_SKIP_{skip};")
            lines.append(
                f"cpu_trace_pending_skip_write(cpu, 0x{block_pc24:06X}, "
                f"(uint8)RECOMP_RETURN_SKIP_{skip}, \"{func_name}\");"
            )

        # Block didn't end with a control-flow op. Emit the explicit edge
        # to its lone CFG successor (linear fall-through) — never rely on
        # textual fall-through into whatever block_order put next, which
        # may have already been emitted earlier in DFS order. Without
        # this, e.g. L_809F's "fall through to L_80A0" silently became
        # "fall through to the function epilogue" for any block whose
        # successor was visited first.
        if not block_terminated:
            succs = block.successors
            blk_pc24 = (bank << 16) | (key.pc & 0xFFFF)
            if len(succs) == 1:
                lines.append(_goto_or_return(succs[0], source_pc24=blk_pc24)
                             + " /* implicit fall-through */")
            elif (len(succs) > 1 and pairs
                  and pairs[-1][0].mnem in ('JSR', 'JSL')):
                lines.append("switch (((cpu->m_flag & 1) << 1) | (cpu->x_flag & 1)) {")
                seen_mx = set()
                fallback_stmt = None
                for succ in succs:
                    mx = (succ.m & 1, succ.x & 1)
                    if mx in seen_mx:
                        continue
                    seen_mx.add(mx)
                    stmt = _goto_or_return(succ, source_pc24=blk_pc24)
                    if fallback_stmt is None:
                        fallback_stmt = stmt
                    idx = (mx[0] << 1) | mx[1]
                    lines.append(
                        f"  case {idx}: {stmt} /* dynamic post-call M{mx[0]}X{mx[1]} */")
                if fallback_stmt is not None:
                    lines.append(f"  default: {fallback_stmt}")
                lines.append("}")
            elif len(succs) > 1:
                lines.append(_goto_or_return(succs[0], source_pc24=blk_pc24)
                             + " /* implicit fall-through */")
            else:
                lines.append("return RECOMP_RETURN_NORMAL; /* no terminator, no successor */")
        block_lines[key] = lines

    # Compose the function source with labels per block.
    src: List[str] = []
    src.append(f"RecompReturn {func_name}(CpuState *cpu) {{")
    # Diagnostics — same call-stack plumbing v1 emitted, so the runtime
    # debug_server's `call_stack` cmd and crash-handler attribution work.
    src.append(f'  extern const char *g_last_recomp_func;')
    src.append(f'  g_last_recomp_func = "{func_name}";')
    src.append(f'  RecompStackPush("{func_name}");')
    src.append(f'  cpu_dbg_funcname("{func_name}");')
    # Trace ring: function entry (carries name hash) — first entry per call.
    fn_entry_pc = (bank << 16) | (start & 0xFFFF)
    src.append(f'  cpu_trace_func_entry(cpu, 0x{fn_entry_pc:06X}, "{func_name}");')
    # Invariant check (AR_MXCHECK): this variant was emitted ASSUMING entry
    # m={entry_m},x={entry_x}. Dispatched calls always match (the switch picks
    # by runtime flags); a DIRECT call here bakes in the emitter's static
    # m/x analysis, so a runtime mismatch means that analysis was wrong — the
    # exact "m/x leak / wrong-variant" bug class — caught at its origin.
    # Near-free when AR_MXCHECK is unset (one global load + branch).
    src.append(
        f'  ar_entry_mx_check(cpu, {entry_m & 1}, {entry_x & 1}, '
        f'"{func_name}", 0x{fn_entry_pc:06X});')
    # Garbage-variant dispatch trap: this variant's decode is a split-immediate
    # misdecode (BRK at ${0:04X} is mid-instruction in a valid sibling). Entering
    # it means a leaked m/x flag dispatched here — fire the trap at the exact
    # entry so the leak is caught at/near its root (DEBUG.md "garbage-variant
    # trap"). Default-on, non-fatal, deduped.
    if _garbage_brk_pc is not None:
        src.append(
            f'  ar_garbage_variant_trap(cpu, "{func_name}", 0x{fn_entry_pc:06X});'
            f'  /* split-immediate BRK at ${_garbage_brk_pc:04X} */')
    # Function-local NLR pending-skip — NOT cpu state. NLR-pattern blocks
    # set this before fall-through to the Return-terminated successor;
    # the Return op reads + clears it. Local-scoped so:
    #   1. NLR signaling is C control-flow state, not 65816 hardware
    #      state — no reason to keep it on CpuState.
    #   2. The optimizer can keep it in a register; no aliasing
    #      concerns through the cpu pointer.
    #   3. Different generated functions can't see each other's
    #      in-flight NLR state.
    # Preceded by a `(void)` cast so the C compiler doesn't warn when
    # NLR detection didn't fire on this function (most functions).
    src.append(f'  RecompReturn _pending_skip = RECOMP_RETURN_NORMAL;')
    src.append(f'  (void)_pending_skip;  /* unused if no NLR site in this fn */')
    # cpu->S balance marker for the PEI-trampoline detector (2026-05-24,
    # narrow variant). Captured at function entry — codegen._emit_return
    # consults it when a Return op's source_pc24 is in _TRAMPOLINE_RETURNS.
    #
    # ALWAYS emitted, not gated on this variant's local `has_pei`: the
    # trampoline-flag set is cross-variant (a Return pc24 flagged by
    # variant M0X0 also fires for M1X1's emit of the same RTS), and
    # inline-cross-fn-blocks can drag a flagged RTS into a function
    # whose own CFG has no PEI. Conditional `_entry_s` left those
    # variants with an undeclared-identifier error at the trampoline
    # branch (mmx_08_v2.c bank-08 build break, 2026-05-24).
    if entry_s_offset:
        sign = '+' if entry_s_offset > 0 else '-'
        src.append(
            f'  uint16 _entry_s = (uint16)(cpu->S {sign} {abs(entry_s_offset)}u);'
            f'  /* entry_s_offset:{entry_s_offset} — caller left stack imbalanced */')
    else:
        src.append(f'  uint16 _entry_s = cpu->S;')
    # Option-1 cpu->S return-frame ABI (see IMPROVEMENTS.md): capture whether
    # a paired host-C caller exists at entry. RTS/RTL may host-return NORMAL
    # only when _hrv==1 AND the stack is balanced (cpu->S == _entry_s);
    # otherwise it dispatches on the popped PC. The caller sets
    # cpu->host_return_valid right before each invoke (direct call -> 1;
    # tail JMP/JML -> propagate the caller's _hrv; dispatch -> 0).
    src.append(f'  uint8 _hrv = cpu->host_return_valid;')
    src.append(f'  if (cpu_take_tailcall_return_context(&_entry_s, &_hrv)) {{')
    src.append(f'    cpu->host_return_valid = _hrv;')
    src.append(f'  }}')
    src.append(f'  (void)_entry_s;  /* used by trampoline balance check */')
    src.append(f'  (void)_hrv;')
    # Record this frame's entry-S parallel to the recomp call stack so a
    # return-to-ancestor RTS (manual PLA/PLX/PLB rebalance + RTS) can be
    # resolved to a SKIP_N non-local return (cpu_resolve_ancestor_skip).
    # Index by the just-pushed g_recomp_stack_top; pop is implicit (top--).
    src.append(f'  if (g_recomp_stack_top >= 1) {{ g_cpu_entry_s[g_recomp_stack_top - 1] = _entry_s; g_cpu_entry_hrv[g_recomp_stack_top - 1] = _hrv; }}')
    for i, key in enumerate(block_order):
        src.append(f"  {_label_for(key)}:")
        # Trace block entry — gives us the SNES PC chain in the trace ring.
        block_pc24 = (bank << 16) | (key.pc & 0xFFFF)
        src.append(f'    cpu_trace_block(cpu, 0x{block_pc24:06X});')
        # Watchdog: per-block heartbeat so tight inner loops trip the 5s
        # frame timeout instead of freezing the runtime indefinitely.
        # Cheap (counter bump + branch); v1 emitted at loop headers, v2
        # gets it at every block since we don't yet identify back-edges.
        src.append(f'    WatchdogCheck();')
        for ln in block_lines[key]:
            # Inject RecompStackPop before any return so the stack stays balanced.
            stripped = ln.strip()
            if stripped.startswith("return"):
                src.append(f"    RecompStackPop();")
            src.append(f"    {ln}")
    # Defensive trailing return so a missing terminator doesn't fall off
    # the end of the function in the C compiler's view.
    src.append("  RecompStackPop();")
    src.append("  return RECOMP_RETURN_NORMAL;")
    src.append("}")
    # hle_func: the decoded body above ran purely for its emission side
    # effects (call/goto target registration for auto-promote, metadata);
    # what lands in the bank source is the forwarding stub.
    if hle_stub_src is not None:
        return hle_stub_src
    return "\n".join(src) + "\n"


def _reg_for_flag(flag) -> str:
    """Helper duplicated from codegen for the local cond-branch rewrite."""
    from v2.codegen import _reg
    return _reg(flag)

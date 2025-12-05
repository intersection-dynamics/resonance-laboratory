# patch_precipitating_event_for_snapshots.py
# Adds snapshot-saving capability to precipitating_event.py automatically.
# Creates precipitating_event.py.bak before modifying anything.

import os, re, sys, shutil

TARGET = "precipitating_event.py"

def die(msg):
    print(f"[PATCH] {msg}")
    sys.exit(1)

def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def ensure_backup(path):
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copyfile(path, bak)
        print(f"[PATCH] Backup created: {bak}")
    else:
        print(f"[PATCH] Backup already exists: {bak}")
    return bak

def insert_after(pattern, insert_text, src):
    m = re.search(pattern, src, flags=re.DOTALL)
    if not m:
        return None
    idx = m.end()
    return src[:idx] + insert_text + src[idx:]

def insert_before(pattern, insert_text, src):
    m = re.search(pattern, src, flags=re.DOTALL)
    if not m:
        return None
    idx = m.start()
    return src[:idx] + insert_text + src[idx:]

def main():
    here = os.getcwd()
    target_path = os.path.join(here, TARGET)
    if not os.path.exists(target_path):
        die(f"Cannot find {TARGET} in: {here}")

    original = read(target_path)
    backup = ensure_backup(target_path)
    text = original

    # 1) Add CLI flags after at least one existing add_argument
    #    We try to find the argument parser block by looking for a line that adds --z-threshold
    cli_anchor_pat = r'(p\.add_argument\([^\n]*--z-threshold[^\n]*\)\s*)'
    cli_insert = r'''
# --- Snapshot saving options (auto-patched) ---
p.add_argument(
    "--snapshot-indices",
    type=str,
    default="",
    help=(
        "Comma-separated list of integer time indices at which to save "
        "the full wavefunction into snapshots.npz (e.g. 50,75,100)."
    ),
)
p.add_argument(
    "--snapshot-every",
    type=int,
    default=0,
    help=(
        "If >0, also save snapshots every K steps (example: --snapshot-every 10)."
    ),
)
'''
    new_text = insert_after(cli_anchor_pat, cli_insert, text)
    if new_text is None:
        # fallback: insert after first argparse.ArgumentParser creation
        cli_anchor_pat2 = r'(p\s*=\s*argparse\.ArgumentParser\([^\)]*\)\s*)'
        new_text = insert_after(cli_anchor_pat2, cli_insert, text)
    if new_text is None:
        write(target_path, original)
        die("Could not locate argparse block to add snapshot CLI flags. No changes made.")
    text = new_text

    # 2) Parse snapshot args right after args = p.parse_args()
    parse_anchor_pat = r'(args\s*=\s*p\.parse_args\(\)\s*)'
    parse_insert = r'''
# --- Snapshot settings (auto-patched) ---
snapshot_indices = set()
if getattr(args, "snapshot_indices", ""):
    for _tok in args.snapshot_indices.split(","):
        _tok = _tok.strip()
        if _tok:
            try:
                snapshot_indices.add(int(_tok))
            except ValueError:
                pass
snapshot_every = 0
try:
    snapshot_every = max(0, int(getattr(args, "snapshot_every", 0)))
except Exception:
    snapshot_every = 0
snapshot_states = {}  # step_index -> numpy array of wavefunction
'''
    new_text = insert_after(parse_anchor_pat, parse_insert, text)
    if new_text is None:
        write(target_path, original)
        die("Could not locate 'args = p.parse_args()' to parse snapshot flags. No changes made.")
    text = new_text

    # 3) Insert a small helper to snapshot 'psi' safely (handles CUDA/no CUDA)
    #    Put it before main() definition if possible.
    helper_pat = r'(def\s+main\s*\()\s*'
    helper_insert = r'''
# --- Snapshot helper (auto-patched) ---
def _save_snapshot_if_needed(step_idx, psi, snapshot_states, snapshot_indices, snapshot_every, use_cuda):
    take = False
    if step_idx in snapshot_indices:
        take = True
    if snapshot_every > 0 and (step_idx % snapshot_every == 0):
        take = True
    if not take:
        return
    import numpy as _np
    try:
        import cupy as _cp  # may fail if not using CUDA
    except Exception:
        _cp = None
    if use_cuda and _cp is not None:
        psi_np = _cp.asnumpy(psi)
    else:
        psi_np = _np.array(psi, copy=True)
    snapshot_states[int(step_idx)] = psi_np
'''
    new_text = insert_before(helper_pat, helper_insert, text)
    if new_text is None:
        # If we can't find def main, append helper at end (harmless)
        new_text = text + "\n" + helper_insert + "\n"
    text = new_text

    # 4) Try to inject calls inside the evolution loop.
    #    We look for a line like: for step_idx, t in enumerate(times):
    loop_pat = r'for\s+([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s+in\s+enumerate\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*:'
    loop_match = re.search(loop_pat, text)
    if not loop_match:
        write(target_path, original)
        die("Could not locate the time-evolution loop (for step_idx, t in enumerate(times):). No changes made.")
    loop_header = loop_match.group(0)
    step_var = loop_match.group(1)
    t_var = loop_match.group(2)
    times_var = loop_match.group(3)

    # We inject a call to _save_snapshot_if_needed(...) right after occurrences of 'psi ='
    # within the loop block. To keep it simple, insert once after the loop header.
    # Users who update psi multiple times per step will still get the last psi captured if we place near end.

    # Find the loop block indent
    loop_start = loop_match.start()
    # Determine current indentation
    line_start = text.rfind("\n", 0, loop_start) + 1
    indent = re.match(r'(\s*)', text[line_start:]).group(1)
    loop_body_indent = indent + "    "

    snapshot_call = f'\n{loop_body_indent}_save_snapshot_if_needed({step_var}, psi, snapshot_states, snapshot_indices, snapshot_every, use_cuda)\n'

    # Insert snapshot call just after the loop header line
    insert_pos = text.find("\n", loop_match.end()) + 1
    text = text[:insert_pos] + snapshot_call + text[insert_pos:]

    # 5) At the end of main(), save snapshots.npz in the run_root
    # We search for a line that says "Evolution complete." print/log and insert after it,
    # otherwise we append near the end of main() by finding the return / end of def.
    tail_insert = r'''
    # --- Save snapshots (auto-patched) ---
    if snapshot_states:
        import numpy as _np, os as _os
        out_path = _os.path.join(run_root, "snapshots.npz")
        _indices = sorted(snapshot_states.keys())
        _psi = [_np.asarray(snapshot_states[k]) for k in _indices]
        _psi = _np.stack(_psi, axis=0)
        _np.savez_compressed(out_path, indices=_np.array(_indices, dtype=int), psi=_psi)
        print(f"[snapshots] Saved {len(_indices)} wavefunction snapshots to {out_path}")
    else:
        print("[snapshots] No snapshots were requested.")
'''
    # Try to insert after a likely "Evolution complete." print
    evo_pat = r'["\']Evolution complete\.["\']\)\s*'
    new_text = insert_after(evo_pat, tail_insert, text)
    if new_text is None:
        # Fallback: insert before "==== Precipitating Event Summary ====" printing
        summary_pat = r'====\s*Precipitating Event Summary\s*===='
        new_text = insert_before(summary_pat, tail_insert, text)
    if new_text is None:
        # Last fallback: append at end of file (within main indentation may not be perfect, but harmless)
        text = text + "\n" + tail_insert + "\n"
    else:
        text = new_text

    # Done
    write(target_path, text)
    print("[PATCH] Success! Snapshot support injected into precipitating_event.py")
    print("[PATCH] Try a test run, e.g.:")
    print('       python precipitating_event.py --geometry lr_embedding_3d_12.npz --output-root outputs --tag test_snap --seed 117 --J-coupling 2.0 --h-field 0.2 --defrag-hot 0.3 --defrag-cold 1.0 --t-total 10.0 --n-steps 101 --z-threshold 0.02 --snapshot-indices 60,80')

if __name__ == "__main__":
    main()

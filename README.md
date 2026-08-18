# dgp-quality-scoring

A data-generating process for a binary `Heart_Disease_Risk` cohort with independently
controllable prevalence and Bayes-optimal AUC, a coupling knob trading off two variable
groups in the label logit, and a layer of hard logical rules that must never be violated
on real data.

## Layout

```
dgp.py                      the DGP: generate(), assert_valid(), solve_binding_rates()
tests/test_dgp.py           the S0 suite
tests/test_differential.py  agreement against reference/dgp_frozen.py
reference/dgp_frozen.py     prior implementation, read-only; imported only by the
                            differential test and its export script
scripts/                    result exports; every artifact in results/ is produced here
results/                    exported CSVs -- the auditable numbers
manifests/                  one run manifest per artifact, plus the stage manifest
```

## Running

```bash
uv sync
uv run pytest tests/ -q
uv run python scripts/export_all.py     # all exports in one process
```

Individual exports (`scripts/export_s0.py`, `scripts/export_differential.py`) can be run
on their own, but use `export_all.py` for the provenance pass below: provenance is stamped
once per process, so running the scripts in sequence would let each one dirty the tree for
the next and only the first manifest would carry a clean stamp.

## Conventions

### Numbers live in artifacts, not in prose

Any figure quoted in a docstring, comment or commit message must be derivable from a
committed artifact under `results/`. Hand-copied numbers go stale silently the moment the
code that produced them changes, and a stale number in a docstring is indistinguishable
from a correct one. Prose either derives the number or points at the artifact and quotes
none. Measurements that depend on a draw record the `n_samples` and `seed` they were taken
at, because level masses and binding rates are seed-dependent.

### Manifests and the two-commit pattern

Every export writes a manifest recording `git_sha_at_run` (HEAD at run time) and
`git_dirty`. A run necessarily predates the commit containing its own outputs, so the
first commit's manifests are always dirty. The pattern is therefore:

1. Commit code and results. The manifests in this commit carry `git_dirty: true` and the
   SHA of the *previous* commit.
2. Re-run `scripts/export_all.py` against the now-clean tree and commit the regenerated
   manifests as a second commit.

After step 2 the published manifest carries a clean SHA that actually contains the code
that produced it. Both commits are kept: the first is the run, the second is its
provenance. Do not try to collapse this into one commit -- a manifest cannot contain the
hash of the commit that contains it.

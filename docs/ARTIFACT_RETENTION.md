# Artifact Retention Policy

## Canonical publication

Generated research outputs are published only when a workflow needs them for the
dashboard, accumulating evidence ledger, or reproducibility manifest. New consumers
should use the canonical paths listed in `README.md`; they must not introduce another
copy of the same payload under a second artifact family.

`artifacts_manifest.json` records SHA-256 hashes for the small set of public status,
governance, execution, and attribution artifacts. Identical hashes across paths are a
signal to consolidate future readers onto one canonical file.

## Large history cleanup

The repository's large historical object database and already-committed duplicates
cannot be made small by deleting files in a normal commit. Correcting that requires a
coordinated Git-history rewrite, force-push, and fresh clones for every collaborator.
That destructive maintenance operation is intentionally outside this hardening change.

Before any later cleanup:

1. Tag and archive the current default branch.
2. Inventory canonical artifacts and verify their hashes.
3. Agree on retention duration and whether Git LFS or release assets will hold archives.
4. Rewrite only the enumerated generated paths.
5. Re-run CI and the publication workflow from a fresh clone before reopening writes.

Until then, workflows avoid adding new duplicate artifact families and publish a
manifest so growth and duplication are observable.

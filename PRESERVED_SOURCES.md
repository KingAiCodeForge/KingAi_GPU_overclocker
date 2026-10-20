# Preserved public source

This candidate starts at public `KingAiCodeForge/KingAi_GPU_overclocker`
`main` commit `4ca16a5ecd541710c94ddb00e39a78c0dd1fce92`. There were no
other remote branches, pull requests, or releases at the initial inspection.
No merge from an invented branch or private repository is claimed.

The original `main` is retained as `preserve/main-20261020`; its complete history
is also recorded in `original-refs.bundle` in the task evidence directory.
The unchanged original clone and separate integration worktree are retained.
Original historical commits `b810041` (logging) and `f9449e3` (initial toolkit)
remain reachable. Retrieve any original script with `git show <commit>:<path>`;
there is no need to duplicate old runtime code in the package.

Only this public repository and its declared package dependencies were used.
No profile backups, real device identities, user logs, probe data, credentials,
private sources, or recommended GPU settings are included in release artifacts.

## Exposure and ignore review

No `AGENTS.md` or `.agents/skills` instructions were present. `.git/info/exclude`
contained Git's example comments only; no `core.excludesfile` was configured.
The existing `.gitignore` excludes environments, builds, caches, logs, `.env`,
`ignore/`, and `tests/probe_*`. No ignored source candidate was present in this
clean public clone. Nothing was force-added or imported from elsewhere, and no
global ignore setting changed. The tracked source allowlist is the package,
tests, workflow, build metadata, license and public documentation.

Wheel contents must contain only `kingai_gpu/` and package metadata/license.
The source distribution also contains synthetic tests and review documentation.
Release artifacts are built from the exact reviewed commit, with an external
receipt and SHA-256 file; task environments and preservation bundles are kept
out of those runtime artifacts.

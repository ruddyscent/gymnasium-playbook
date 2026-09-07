# Project development guidelines

## Educational implementations

- Make core equations and algorithm steps explicit. Avoid unnecessary abstraction
  and dependencies; preserve the repository's educational focus.
- Write mathematical formulas in LaTeX in documentation and explanations. In
  Markdown, use `$...$` for inline math and `$$...$$` for display equations.
- Place implementations under `environments/<category>/<environment>/`. Never
  create a local `gymnasium` package that shadows the installed library.
- Separate learning, read-only evaluation, and visualization.
- Include type annotations. For code changes, run relevant regression tests and
  static type checks, covering semantic boundaries such as termination versus
  truncation. Report the checked scope and any unverified code.

## Reproducible environments

- Prefer common packages and the same Python and dependency versions across
  Windows x86-64, macOS on Apple Silicon (ARM64), and Ubuntu x86-64 wherever
  practical. These are the full-stack targets; make other architecture
  assumptions explicit.
- Manage dependencies with `uv` and maintain the shared `uv.lock`. Read
  `.python-version`, `pyproject.toml`, and `uv.lock` for current versions and
  constraints instead of duplicating them here. Use locked installation and
  execution commands documented in the root README.
- Before adding or upgrading dependencies, check upstream Python support and
  wheel availability for each intended OS/architecture combination.
- Isolate unavoidable platform-specific dependencies or GPU backends with
  optional dependencies or explicit platform markers. Document the reason,
  supported platforms, and available fallback (or its absence); never silently
  exclude a target platform.
- Avoid machine-specific paths or globally installed tools as implicit
  prerequisites. Document portable commands, required tool setup, and unavoidable
  native libraries, display requirements, or graphics backends in the relevant README.
- Validate installation and relevant runtime behavior on all three target
  platforms through `.github/workflows/`. Successful lockfile resolution or
  available wheels alone do not prove runtime compatibility. Distinguish local
  checks from CI results and name any unverified platform or feature, especially
  GUI rendering; headless smoke tests do not verify interactive graphics.
- Record seeds, hyperparameters, environment configuration, software versions,
  and evaluation procedures. Save generated models and run artifacts under
  ignored paths such as `runs/`.

## Issue and delivery workflow

- Read the issue and discussion before implementing it. Use a dedicated working
  branch and worktree, preserving unrelated user changes.
- When work begins, register the working branch in the issue's GitHub Development
  section and verify that the connection was saved. Link the implementation PR
  there as well and verify it. A branch name in prose or a PR mentioning the
  issue is not a verified Development connection.
- If direct branch linking is unavailable, explicitly report the limitation and
  record the branch URL and linked PR when available. If push or PR publication
  is awaiting authorization, report the pending connection rather than claiming
  registration succeeded; complete and verify it when authorized.
- A request to work on an issue authorizes routine progress and substantive
  implementation-insight comments on that issue, including measured results and
  limitations. It does not authorize blog publication or unrelated messages.
- Keep executable usage and verification instructions in the implementation
  README. Maintain detailed explanations and blog drafts on the separate `blog`
  branch. Keep task history, commit IDs, and measured results in issues or
  documentation, not in permanent agent instructions.
- Commit, push, and merge only when explicitly authorized. Preserve configured
  commit signing and verify the signing mechanism before committing; diagnose
  signing failures without bypassing signing. Use focused Conventional Commits
  with explanatory bodies.
- Before requesting or performing a merge, review every issue acceptance
  criterion against the implementation, documented commands, tests, and recorded
  results. Update the issue checklist and record supporting evidence. Check only
  verified criteria; leave unmet or unverified items unchecked with an explanation.
  Passing CI or a merged PR alone does not establish completion.
- Before an authorized squash merge through a PR, verify the relevant CI jobs on
  Windows, Apple Silicon macOS, and Ubuntu for the PR's final revision. Respect
  the current `main` ruleset, including PR review-thread resolution and linear
  history; do not bypass protections.
- After merging, verify the issue's Development links, acceptance checklist, and
  completion evidence against the final result. Record remaining work and
  verification limits. Close the issue only when explicitly authorized;
  otherwise report that it remains open.

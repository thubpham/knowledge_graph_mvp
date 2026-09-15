---
name: repo-audit
description: Audit the repo for dead/orphaned files, stale artifacts, and repo-hygiene issues (untracked bloat, misplaced tests, half-wired modules) before a commit. Use when the user asks to "clean up the repo", "audit what's unused", "what can we remove before committing", or before committing a large batch of changes with many new/untracked files. Read-only investigation + a report; nothing is deleted or moved without explicit user confirmation per item.
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
  - Edit
  - AskUserQuestion
---

# /repo-audit — Pre-commit repo hygiene sweep

Finds candidates for removal/relocation before a commit, without guessing —
every non-obvious call (delete vs. keep, wire in vs. leave stubbed) goes back
to the user as a decision, not an assumption. This exists because "clean up
before committing" silently turned into either (a) deleting something still
in active use, or (b) leaving genuinely dead weight in the tree — both are
worse than asking.

## Step 1 — Inventory

```bash
git status --short
git ls-files | wc -l
find . -maxdepth 2 -type d -not -path './.git*' | sort
```

Separate untracked new files/dirs from modified tracked ones. New,
untracked top-level additions are the highest-signal audit targets — that's
usually where scope crept in.

## Step 2 — Classify every new/suspicious file or folder

For each candidate, determine real usage, don't guess from the name:

- **Docs/generated content** (e.g. an `openwiki/`-style folder): check the
  project's CLAUDE.md / AGENTS.md for whether it's described as
  auto-generated, and check `.github/workflows/` for the automation that
  owns it. Auto-generated + workflow-owned = keep, don't hand-edit.
- **Entry points** (servers, CLI scripts): grep for imports across the repo,
  but absence of in-repo references doesn't always mean dead — an MCP
  server or subprocess entrypoint is invoked from *outside* the repo
  (Claude Desktop config, cron, CI). Check README/AGENTS.md for how it's
  actually launched before flagging it as orphaned.
- **Modules under source directories** (e.g. `enrichment/`, `core/`): for
  each file, `grep -rl` its module/function names across the rest of the
  repo. Zero referencing files is the real signal — but check whether it's
  a recently-added feature not yet wired into its entrypoint (e.g. a new
  ingester not imported by the main ingest script) rather than genuinely
  dead code. That's a "needs decision" case, not an auto-remove: there may
  be a deliberate reason (see Step 3).
- **Loose test files at repo root**: check if they still pass
  (`python <file>`), whether a `tests/` convention already exists, and
  whether CI references them.
- **Large binaries / reference material** (PDFs, datasets): check
  `.gitignore` and `git ls-files` to see if they're already tracked, and
  whether anything in code/docs references them. Growing untracked binary
  content is a repo-bloat flag even if nothing is "broken."
- **`.claude/`, `.vscode/`, `__pycache__/`, `.local/`-style dirs**: confirm
  gitignore status before treating a "new untracked" report as a hygiene
  issue — most of this is already ignored and not really a commit concern.

## Step 3 — Don't assume "unused" means "safe to wire in" either

If something looks unused because it was never wired into its caller,
check *why* before assuming it's a forgotten TODO. Read what the module
actually does — if it touches something sensitive (raw personal data,
external subprocess calls, anything self-referential like ingesting the
tool's own logs/transcripts), the omission may be deliberate risk avoidance,
not an oversight. Surface the specific risk you found, then ask.

## Step 4 — Report, then ask, then act

1. Give a concise structured report: keep / needs-decision / remove-candidate,
   with one-line evidence per item. Don't editorialize beyond the evidence.
2. For every needs-decision item, use `AskUserQuestion` — one question per
   item, not a single bundled yes/to-all. Include the concrete evidence in
   the option descriptions so the user isn't deciding blind.
3. Only after answers come back, make the edits (gitignore entries, file
   moves, explanatory comments, wiring). Never delete a tracked file or
   rewrite git history as part of this — moves/ignores only, and only for
   items the user explicitly confirmed.
4. Re-run `git status --short` at the end and confirm the tree looks clean
   before telling the user it's ready to commit.

Do not use this skill to actually create the commit — that's a separate,
explicit request per this repo's standing git-safety rules.

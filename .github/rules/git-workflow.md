# Git & Branching Workflow for AI Agents & Developers

This document defines the branching, commit, and pull request policies for the Barrikade repository. All developers and **AI agents** (Claude Code, Cursor, Copilot, Antigravity) must adhere to these policies.

## Branch Naming Conventions

Always name your working branches using the following format:
```
<type>/[issue-number-]<short-kebab-case-description>
```

### Types
- **`feature/`**: For new features or SDK expansion.
- **`fix/`**: For bug fixes.
- **`chore/`**: For internal configs, chores, and helper updates.
- **`docs/`**: For documentation-only changes.
- **`test/`**: For test additions/refactoring.

### Examples
- **No issue**: `feature/add-baseline-detector`
- **Linked to issue #142**: `fix/142-handle-empty-input`
- **Chore task**: `chore/configure-ruff-linting`

---

## Commit Guidelines

### 1. Atomic Commits
- Break your work down into **atomic logical units**.
- Each commit should stand on its own, compile, and be reviewable in isolation. Avoid large "kitchen sink" commits.

### 2. Sign-offs (Developer Certificate of Origin)
- Sign off every commit you create: `git commit -s`. This appends a `Signed-off-by: Your Name <email>` trailer to verify you have the right to submit this code.
- If you are an AI agent committing via Git tools, automatically append this trailer or use `git commit -s`.

### 3. Conventional Formats
- The commit header MUST follow the Conventional Commits template. Refer to `conventional-commits.md` for specific rules.

---

## Pull Requests

### 1. Small and Focused
- Keep PRs small. If a task requires changes in multiple unrelated areas, split them into multiple PRs.

### 2. PR Body Links
- Search existing issues and link them in your PR **body** (e.g., `Closes #123` or `Refs #142`).
- Do NOT mention issue numbers in the PR title, only the body.

### 3. Squash Merging
- All pull requests in this repository are **squash-merged** onto the main branch.
- The squash commit message MUST follow Conventional Commits format, utilizing the PR title as the squash header and the PR description/commits as the body.

---

## Issue Tracking

If you (or an AI agent) discover an issue, bug, or improvement during development:
1. **Verify first**: Re-read the source code or run tests to prove the bug exists.
2. **Search existing issues**: Avoid duplicate tracking. Search via GitHub web UI or CLI: `gh issue list --search "query"`.
3. **Report**: Offer to document it. If an agent is working, it should never create a GitHub issue automatically unless the human user explicitly instructs it to do so.

## Branches

- Prefix branches by change type: `feature/`, `fix/`, `chore/`, `docs/`, `test/`.
- After the slash, use a short kebab-case description: `fix/handle-empty-input`.
- If an issue exists, include its number: `fix/142-handle-empty-input`.

## Commits

- Break work into atomic logical units—each commit should stand on its own and be reviewable in isolation.
- Reference related issues with `Refs #<issue>` in the commit body.
- Do NOT use GitHub magic close-keywords (`Closes`, `Fixes`, `Resolves`) in commit messages—those belong in the PR body so they close issues on merge, not on every push.
- Sign off every commit per the Developer Certificate of Origin: `git commit -s` (adds a `Signed-off-by:` trailer).

## Issues

If you spot something worth tracking outside the current task, surface it and offer to file a GitHub issue—never create one yourself unless the user explicitly asks. Before creating any issue:
- Verify the claim (re-read the code, run the relevant test, reproduce if possible).
- Search existing issues (`gh issue list --search "..."`) to avoid duplicates.

## Pull requests

- Keep PRs small and focused. Split unrelated changes into separate PRs.
- Link touched issues in the PR body.
- PRs are squash-merged. The squash commit message must follow Conventional Commits (see `conventional-commits.md`).

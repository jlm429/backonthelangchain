# Agent Guidelines

Minimal repository guidance for coding agents. Inspired by Kung Chen's GitHub
agent guidance https://github.com/kunchenguid 

## Engineering Standard

- Apply a high standard of engineering excellence.
- Keep changes small, direct, and reviewable.
- Do not use em dashes in prose, code comments, docs, commit messages, or PR text.
- Do not add an agent name as a commit co-author.
- If you encounter lint failures, test failures, or test flakiness, fix them even
  when they are not caused by the current task.
- For bug fixes, start by reproducing the issue in an end-to-end setting so the
  real failure is understood before changing code.
- If there is a UI, be exacting about interaction, layout, copy, spacing,
  responsiveness, accessibility, and visual polish.

## Security

- Never read, print, summarize, copy, commit, or expose `.env`, `.env.*`, API
  keys, tokens, credentials, or service account files.
- Document required secrets only by variable name in `.env.example`.
- If a secret is exposed, stop and alert the human reviewer.

# Changelog

Notable changes to this project. Entries are one-liners phrased as what
changed for a user of the chat, not as which files moved; the
fine-grained record is the git history. Entries accumulate under
Unreleased while work happens on a branch; just before the branch merges
to main, Unreleased is retitled to its merge date plus a short title,
and merged sections are never edited again. Format loosely follows
[Keep a Changelog](https://keepachangelog.com).

## [Unreleased]

### Added
- Chat engine, tool registry, and the project's first test suite in
  `core/` — one engine now drives both frontends.

### Changed
- Terminal UI is now a thin renderer over the shared engine; CLI flags
  and behavior unchanged.
- Web UI is now a thin renderer over the shared engine, like the
  terminal UI before it.
- Terminal chats start faster: the embedding model now loads only when
  a PDF is actually passed.

### Fixed
- `--rag-tool` with a bad path printed "PDF file not found: None"; it
  now prints the actual path.
- Hitting the tool-call budget no longer fires one wasted model request
  before forcing the final answer.

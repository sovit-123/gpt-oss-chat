# Changelog

Notable changes to this project. Entries are one-liners phrased as what
changed for a user of the chat, not as which files moved; the
fine-grained record is the git history. Entries accumulate under
Unreleased while work happens on a branch; just before the branch merges
to main, Unreleased is retitled to its merge date plus a short title,
and merged sections are never edited again. Format loosely follows
[Keep a Changelog](https://keepachangelog.com).

## 30 September, 2026 - Core Engine Refactor

### Added
- Chat engine, tool registry, and the project's first test suite in
  `core/` — one engine now drives both frontends.

### Changed
- Terminal UI is now a thin renderer over the shared engine; CLI flags
  and behavior unchanged.
- Web UI is now a thin renderer over the shared engine, like the
  terminal UI before it.
- Both UIs start faster: the embedding model now loads on first document
  query instead of at import time.
- Model requests now time out after 120 seconds instead of the OpenAI
  SDK's ten-minute default, so a stalled stream fails fast.
- Tool functions report failure by raising; the registry converts
  exceptions into feedback for the model, replacing the fragile
  "Error:"-string convention.
- README now documents how to run the tests.

### Removed
- The pre-engine debug script `tools/tool_simulation.py`, the unused
  history helpers in `utils/prompt.py`, and a demo block in
  `semantic_engine.py` that pointed at a file not in the repository.

### Fixed
- `--rag-tool` with a bad path printed "PDF file not found: None"; it
  now prints the actual path.
- Fresh installs could not run the web UI: `gradio` was missing from
  `requirements.txt`. Test dependencies now live in
  `requirements-dev.txt`.
- Hitting the tool-call budget no longer fires one wasted model request
  before forcing the final answer.
- Tools rejected calls that omitted optional arguments such as the
  search engine or `top_k`; the defaults now apply as the schemas
  always claimed.
- The system prompt announced a three-call tool limit while the engine
  allowed five; both now come from one setting.

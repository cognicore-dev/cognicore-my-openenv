# Changelog

## [Unreleased]

### Changed
- Synced contributor docs and package metadata to `cognicore-dev/cognicore-my-openenv`, package version `0.9.5`, and Python 3.10+

## [1.0.0] - 2026-07-06

### Added
- **AgentPassport (`cognicore.passport`)**: Universal serialization wrapper (zip/json/npz) for packaging agents.
- **AgentDNA (`cognicore.dna`)**: Behavioral genome extraction and evolutionary algorithms with SVG visualization.
- **Conscience (`cognicore.conscience`)**: Real-time self-auditing, uncertainty detection, and decision holding.
- **Civilization (`cognicore.civilization`)**: Federated agent behavioral knowledge sharing and server node.
- **TimeTraveler (`cognicore.timetravel`)**: State rewinding, counterfactual branching, and timeline comparison.
- **Oracle (`cognicore.oracle`)**: Predictive simulation, adversarial search, and goal-directed hallucinations via environment models.
- **DreamEngine (`cognicore.dream`)**: Synthetic experience generation, adversarial nightmares, and hallucinatory rollouts.
- **CLI Enhancements**: Added `passport` and `civilization` management commands to the central CLI.

### Changed
- Refactored core modules to prepare for the "Living Agent Stack".


All notable changes to CogniCore NEXUS are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.9.3] — 2026-06-18

### Changed
- **MemoryBackend ABC migration complete** — `get_context()` → `get_by_category()`, `retrieve()` → `get_by_category()` across all backends and callers
- Fixed 15 test failures caused by interface migration
- All **525 tests passing**

### Fixed
- Prompt injection pattern improvements in SafetyMonitor
- Smoothed IDF computation in TFIDFMemoryBackend for stability with small corpora

---

## [0.9.2] — 2026-06-11

### Added
- **PyPI release** with full benchmark report

### Changed
- Memory benchmark results: **12.2% solve rate** vs 1.1% baseline (11× improvement)

---

## [0.9.1] — 2026-06-10

### Fixed
- 4 remaining bug fixes from 0.9.0 release
- Benchmark report corrections and README rewrite

---

## [0.9.0] — 2026-06-09

### Fixed
- 8 memory and reflection bug fixes across `TFIDFMemoryBackend`, `ReflectionEngine`, and `RewardBuilder`

### Changed
- 4 performance optimizations in memory retrieval and reflection scoring
- **TFIDFMemoryBackend** established as the default backend for all runtime paths

---

## [0.8.3] — 2026-06-05

### Fixed
- AutoCurriculum task generation edge cases
- `Experiment.learn()` convergence issues with small episode counts

---

## [0.8.2] — 2026-06-03

### Fixed
- AutoLearner initialization when no prior episodes exist
- SafetyLayer false-positive detection in edge cases
- `episodes=0` parameter handling across train/evaluate APIs

---

## [0.8.1] — 2026-06-01

### Fixed
- Memory retrieval returning stale entries after backend swap
- Replay persistence corruption on interrupted writes
- Branch tracking state desync in multi-agent scenarios

---

## [0.8.0] — 2026-05-28

### Added
- **NEXUS Live Runtime Dashboard** with 9 real subsystems:
  - Memory Explorer, Reflection Viewer, Safety Monitor, Reward Tracker
  - Episode Timeline, Agent Health, Middleware Pipeline, Config Editor, Logs
- **Agent Immune System** (`cognicore/immune/`) — anomaly detection and adaptive response for agent behavior
- **Replay Time Travel** (`cognicore/replay/`) — rewind, branch, and replay agent episodes with RL integration

---

## [0.7.0] — 2026-05-20

### Added
- **NEXUS Autonomous Runner** — Devin-like engine for end-to-end task execution
- **Enterprise integrations:**
  - GitHub (PR creation, issue management)
  - CI pipeline orchestration
  - Slack notifications
  - Linear issue tracking
  - Scheduler for recurring tasks
- **Multi-agent runtime** with 4 routing policies:
  - Round-robin, capability-based, load-balanced, priority-queue

---

## [0.6.0] — 2026-05-10

### Added
- **Research features:**
  - `FailurePredictor` — predict agent failure before it happens
  - `CognitiveMemory` — biologically-inspired memory consolidation
  - `RedVsBlue` — adversarial agent evaluation framework
  - `AIDebugger` — automated agent behavior debugging
  - `IntelligenceScorer` — multi-dimensional agent capability scoring
  - `ThoughtTracer` — agent reasoning chain visualization
  - `KnowledgeTransfer` — cross-agent knowledge sharing
  - `EvolutionEngine` — evolutionary strategy optimization for agents
- **SWE-bench runner** with **20/20 resolve rate** on test suite

---

## [0.5.0] — 2026-05-01

### Added
- **Real-world safety environment** — `RealWorldSafety-v1` with 30 curated cases:
  - Jailbreak attempts (DAN, roleplay, character-based)
  - PII leakage (SSN, credit cards, addresses)
  - Prompt injection (translation, summarization attacks)
  - Hate speech, self-harm, illegal activity detection
  - Medical/legal advice boundary cases
  - Tricky safe edge cases ("kill the process", "fire an employee")
- **Real-world code bugs environment** — `RealWorldCodeBugs-v1` with 18 production Python bugs:
  - SQL injection, race conditions, resource leaks
  - Mutable default arguments, closure bugs, bare excepts
  - Hardcoded secrets, async antipatterns, type errors
- **6 new environment variants** — Easy/Medium/Hard for both real-world envs
- **30 total environments** (up from 24)

### Changed
- Fixed PyPI URLs (previously pointed to non-existent `cognicore/cognicore` repo)
- Added `CHANGELOG.md` with full version history
- Updated roadmap with dated milestones

---

## [0.4.0] — 2026-04-30

### Added
- **Custom error hierarchy** — 7 exception classes with actionable messages and suggestions
  - `InvalidEnvironmentError` — shows similar env names ("Did you mean?")
  - `InvalidConfigError` — catches bad config on construction
  - `AgentInterfaceError` — clear message when agent lacks `act()`
  - `EpisodeFinishedError` — replaces silent failure on double-step
- **AgentProtocol** — runtime-checkable Protocol for duck-typing agent validation
- **Config validation** — `CogniCoreConfig.__post_init__()` validates all fields immediately

### Changed
- **Replaced 300+ `print()` calls with `logging`** across 31 modules
- **Type-safe API** — `train()` and `evaluate()` validate agent/env/episodes before running
- `make()` now raises `InvalidEnvironmentError` instead of generic `KeyError`
- `step()` raises `EpisodeFinishedError` instead of silently returning empty data

---

## [0.3.0] — 2026-04-28

### Added
- **CLI commands** — `cognicore train`, `cognicore demo`, `cognicore metrics`
- **Config-driven training** — YAML config files (`configs/default.yaml`, `configs/strict_safety.yaml`)
- **Deterministic benchmarks** — 5 seeds × 10 episodes, mean ± std dev, saved JSON reports
- **Real-world use case** — `examples/chatbot_safety_eval.py` (chatbot safety evaluation)
- **Learning curve graph** — `docs/learning_curve.png` embedded in README
- **README overhaul** — tagline, before/after output, comparison table, how-it-works diagram
- **Known limitations section** — 5 honest limitations documented
- **Roadmap** — plugin ecosystem vision (cybersec, finance, eval)
- `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`

---

## [0.2.0] — 2026-04-15

### Added
- **24 environments** across 6 domains (safety, math, code, conversation, planning, summarization)
- **`cc.train()` / `cc.evaluate()`** — clean 2-line API
- **22 CLI commands** — `cognicore list`, `run`, `benchmark`, `serve`, `dashboard`, etc.
- **Structured Rewards** — 8-component reward signal per step
- **PROPOSE → Revise protocol** — tentative exploration before commitment
- **Safety Monitor** — streak detection and health status
- **Gymnasium adapter** — `CogniCoreGymAdapter` for RL compatibility
- **API server** — FastAPI-based REST API
- **GitHub Actions CI** — tests on Python 3.9/3.11/3.12, linting, security scan

---

## [0.1.0] — 2026-04-05

### Added
- Initial release
- Core `CogniCoreEnv` base class with Memory, Reflection, and Structured Rewards
- `SafetyClassification-v1` environment
- Basic agent interface (`BaseAgent`, `RandomAgent`)
- `cognicore.make()` factory function

---

[0.9.3]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.8.3...v0.9.0
[0.8.3]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.8.2...v0.8.3
[0.8.2]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Kaushalt2004/cognicore-my-openenv/releases/tag/v0.1.0

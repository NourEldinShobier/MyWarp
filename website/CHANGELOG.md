# Changelog

This document records the key changes in each Zap release. Only functional commits are included; internal rolling tags such as dev / stable are omitted.

## [Unreleased]

- **AI / BYOP**: port opencode `applyCaching` and enable prompt caching; `write_to_long_running_shell_command` rejects embedded LF in line mode; BYOP LRC monitor fallback now uses a silent subtask; fix sender leak in the 50ms window of `cancel_execution` (#134 follow-up, #137)
- **Cloud removal Phase 1–2**: add a `cloud-disabled` channel predicate; remove billing/pricing, referral/reward and cloud sharing dialog UI; unsubscribe the RTC UpdateManager; retire the notebook/folder sync queue
- **Platform**: fix a panic when launching macOS from Spotlight/Finder/Launchpad; `run_shell_command` stdout now falls back to the command grid
- **Infra**: `.gitattributes` enforces LF; add stale bot and Claude Code GitHub workflow
- **Editor**: code/Markdown viewer gains syntax highlighting for 15 languages (Dart, Zig, SCSS, R, Julia, OCaml, Erlang, Nix, Groovy, Solidity, GraphQL, Protobuf, Clojure, Elm, CMake)

## [v2026.05.06.preview] — 2026-05-06

- **AI**
  - Integrate the DeepSeek CLI agent and improve LSP install reliability
  - LSP is now a global `enabled_lsp_servers` setting; remove the `/index` command and the codebase indexing runtime
  - `/plan` faithfully reproduces Plan Mode (system prompt + hard tool guardrails)
  - Agent dynamic tool whitelist, `persist_conversations` setting, `ask_user_question` always asks under auto-approve
  - BYOP supports provider extra headers
- **Fixes**
  - `apply_file_diffs` schema changed from `const` to `enum` for Gemini compatibility
  - Root cause of SSE stalls: genai gzip now off by default + workflow split
  - Plan folder notebooks are created immediately in environments without cloud
- **Branding**: logo and icons use a white background; BYOP mode hides credits/billing UI

## [v2026.05.04.preview] — 2026-05-04

- **SSH Manager**: data layer + persistence + keychain landed; full UI/UX integration (panel + central pane + drag and drop + collapse + Connect + Command Palette)
- **AI**: distinguish "no suggestion" model output and improve the prompt system; BYOP history multimodal support extended to PDF/audio, opencode-style ERROR replacement; UserQuery.context.images kept alive end to end
- **UI**: title bar search box can be hidden; fix keybinding settings edit state and shortcut badge contrast
- **i18n**: translate the remaining fixed strings in the main UI; `/model` default binding `alt-shift-/`
- **Fixes**: Anthropic adapter sends the 1M context beta header by default; BYOP ToolCall emits a placeholder card on the first frame; OpenAI-strict providers no longer echo back `reasoning_content`
- **Infra**: CI fixes the `.deb` build and enables PR tests

## [v2026.05.03.preview(.2/.3/.4)] — 2026-05-03

- **Upstream sync**: merge a large batch of warp-upstream commits (cross-window tab drag, shell script detection, IME cursor, remote server initialization refactor, SSH remote-server auto-upgrade, etc.); set up rerere + the `zap-ours` merge driver; add a blocklist document
- **AI / BYOP**: coerce layer for type-mismatched tool argument output; tighten the suspicious backslash scan to eliminate ls/diff false positives
- **i18n**: complete the Chinese localization (settings panel, etc.)
- **Website**: unify the GitHub URL to `zerx-lab/warp`; fix horizontal overflow on mobile
- **Fixes**: align the Windows taskbar ICO with the upstream format; NLD in terminal defaults to true again so Chinese input goes to AI automatically

## [v2026.05.02.preview] — 2026-05-02

- **AI / BYOP**
  - Complete the conversation compaction loop: `byop_compaction` module, settings persistence, auto prune, overflow passthrough, a 1:1 port of opencode
  - Move reasoning effort from provider settings to the input box picker
  - Multimodal attachment support in the BYOP path
  - Local BYOP webfetch / websearch with Exa integration
  - Select the system prompt template by model identifier; add several templates
- **Privacy / cloud removal**
  - Physically delete easily removable P4 dead code (anonymous_id / EXPERIMENT_ID_HEADER / settings sync / app_focus)
  - Cut four outbound paths: closed-source telemetry, Sentry, anonymous_id, Settings sync
  - Three privacy switches default to false instead of true
  - Two cleanup waves for `cloud_conversations` (UI / privacy / FeatureFlag / AIClient / cargo feature)
- **Refactor**: remove blocklist AI response scoring and its telemetry; remove `agent_attribution` and the Oz changelog toggle
- **CI**: weekly builds become formal releases with standardized tags

## [v2026.05.01.preview] — 2026-05-01

- **Cloud removal**: physically delete 6 cloud LLM tools + child_agent + orchestration; physically delete the three share modal components and the billing denied modal; website switches to a monochrome logo
- **AI**
  - Workflow Autofill uses BYOP one-shot
  - BYOP LRC keeps injecting context in later turns + stronger sanitize + control key tokens
  - Chat stream adds remote login session hints and reasoning passthrough
  - genai error mapping refined into Stream / Other variants
  - chat stream adapter, fix ToolCall None handling
- **Platform**: `warpui_core` avoids rescanning system fonts; sync commands unconditionally disable the pager and use `PAGER=cat` to preserve the real exit code
- **Website**: refactor site-wide components and i18n; sync Tailwind and global styles

## [v2026.04.30.oss] — 2026-04-30

- **CI**: CHANNEL `preview` → `oss`; fix Windows / macOS build failures
- **Refactor**: delete leftover cloud_mode code and settings

## [v2026.04.30.preview] — 2026-04-30

First preview release of the Zap community fork.

- **Branding and positioning**: rename to Zap + new logo + community fork README
- **BYOP**
  - `async-openai` → `genai`, with explicit binding for 5 native protocols
  - Providers sub-page + models.dev data source + quick-add search box
  - Slimmed-down prompt templates
- **Decentralization cleanup**: remove the `UseComputer` / `RequestComputerUse` tools, the Drive `Create team` / `Join team` entries, and referral-related code
- **i18n**: Fluent infrastructure + translation of 12 settings_view files; complete i18n for the ai / features / teams pages
- **Website**: add a BYOP landing page (Astro + Tailwind, bilingual Chinese/English); responsive improvements
- **AI**: CJK input classification, reasoning split, BYOP tool_call diagnostics, LRC tag-in synthesizes a virtual subagent + floating window spawn path
- **CI**: Release explicitly declares `contents: write` permission to fix 403

[Unreleased]: https://github.com/zerx-lab/warp/compare/v2026.05.06.preview...HEAD
[v2026.05.06.preview]: https://github.com/zerx-lab/warp/compare/v2026.05.04.preview...v2026.05.06.preview
[v2026.05.04.preview]: https://github.com/zerx-lab/warp/compare/v2026.05.03.preview.4...v2026.05.04.preview
[v2026.05.03.preview(.2/.3/.4)]: https://github.com/zerx-lab/warp/compare/v2026.05.02.preview...v2026.05.03.preview.4
[v2026.05.02.preview]: https://github.com/zerx-lab/warp/compare/v2026.05.01.preview...v2026.05.02.preview
[v2026.05.01.preview]: https://github.com/zerx-lab/warp/compare/v2026.04.30.oss...v2026.05.01.preview
[v2026.04.30.oss]: https://github.com/zerx-lab/warp/compare/v2026.04.30.preview...v2026.04.30.oss
[v2026.04.30.preview]: https://github.com/zerx-lab/warp/releases/tag/v2026.04.30.preview

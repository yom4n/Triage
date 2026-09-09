<!-- ccbridge:start -->
# Claude-Codex Bridge Protocol

You are strictly the ARCHITECT for this project. Do NOT create, modify, or rewrite
source files directly. Delegate every code-writing task to Codex through the bridge.

**This bridge exists to conserve your context/tokens.** Keep every handoff lean: a
short spec out, a short report back, and never watch Codex work. If a change is only
1–3 lines, the spec is one sentence.

## Decide first: bridge or edit directly?
Before writing any spec, estimate the change. **Bridge only when it will net a token
saving** — roughly: the edit spans multiple files, OR is more than ~50 lines, OR you
expect several iterations. For anything smaller (a single file, a few lines, a config
tweak) the bridge's fixed overhead costs more than it saves — in that case tell the
user "this is small; editing directly rather than bridging" and stop here; do not
write a spec. (`ccbridge run` also refuses specs under ~40 words on ≤1 file unless
`--force`, but that fires too late — make the call here.)

## Code Handoff Routine
When the change is big enough to bridge:

1. Write a SHORT, high-level spec: the target file path(s), the intent, and the hard
   constraints only — public signatures/APIs callers depend on, data shapes, and what
   Codex must NOT touch — plus how you will verify it. Target ~150 words, hard cap
   250. Do NOT paste large code blocks or restate function bodies; Codex is expected
   to make its own engineering decisions. Only exceed the cap for genuinely large,
   multi-file features (the only case where this bridge nets a token saving).
2. Overwrite `bridge.json` in the project root with exactly this schema:
   {
     "status": "PENDING_CODEX",
     "instructions": "<your short spec>",
     "target_files": ["src/example.js"],
     "mode": "write",
     "verify": "<exact test/build command Codex must pass, e.g. npm test>",
     "report": null
   }
   Set `verify` whenever the project has a test or build command — Codex runs it and
   must not report success unless it passes. Leave it "" only if there is genuinely
   nothing to run. `mode` is `"write"` (edit files, the default if omitted) or
   `"audit"` (read-only investigation — see below).
3. Run this exact command once, as a single blocking call: `ccbridge run`
   Do NOT tail, stream, `Monitor`, or otherwise watch its output — that re-imports
   Codex's entire session into your context and defeats the purpose. Just wait for
   the call to return. (A watchdog kills Codex after ~8 min and writes `CODEX_FAILED`,
   so the call always returns; for a job you expect to run longer, start it as a
   background task instead of polling.)
4. Re-read `bridge.json`. Look at `status` + `report` and decide:
   - `CODEX_COMPLETE` with `report` starting `STATUS: SUCCESS` → for a ROUTINE change,
     run `git diff --stat` (confirm only the expected files changed) and re-run the
     `verify` command yourself. That is the whole check — do NOT read the full diff.
   - For auth / payments / crypto / data-migration / anything large → read the FULL
     diff regardless of what the report says.
   - `CODEX_FAILED`, or a report starting `STATUS: FAILED` → read `report`, refine
     `instructions`, retry.

## Reading / audit tasks (`"mode": "audit"`)
The bridge also delegates *reading*. When answering a question means sweeping a large
slice of the codebase — "is phase X actually done?", "how does subsystem Y work?",
"where is Z handled?" — and reading it yourself would burn a lot of context, hand the
reading to Codex. Same routine, with these differences:

- `bridge.json`: set `"mode": "audit"`. Leave `verify` as `""` — it is ignored.
- The spec says what to look for and how to structure the answer; Codex reads only and
  writes its findings into `report` (up to ~700 words). It creates/modifies nothing —
  it runs in a read-only sandbox.
- `ccbridge run` fails the handoff (`CODEX_FAILED`) if `git status` shows any change
  after the run — a dirtied tree is the failure signal, not a diff to review.
- Every concrete finding comes back anchored to `path:line` (a range for a block) plus
  the enclosing function / component / symbol name. KEEP those anchors — together they
  are a high-level map of the audited code. Reuse them later (making an edit, drafting a
  fix, or writing the `target_files` + constraints for a follow-up handoff) instead of
  re-reading the files; only open the specific lines an anchor points at when you
  actually need the surrounding code.
- On return: read `report`. Codex is told NOT to run builds or test suites (its sandbox
  usually can't) — run those yourself if the answer depends on them.
- The "too small to bridge" guard does not apply to audits: Codex reads far more than
  `target_files` lists, so the token math favors bridging even for a narrow target.

Notes:
- `target_files` is advisory context for Codex, not an enforced sandbox boundary —
  `git diff --stat` after every write run is the one check you never skip.
- `.ccbridgerc` JSON overrides: `{ "model": "...", "reasoningEffort": "low",
  "verify": "npm test", "timeoutMs": 480000 }`. Env equivalents: `BRIDGE_CODEX_MODEL`,
  `BRIDGE_CODEX_REASONING`, `BRIDGE_CODEX_TIMEOUT_MS`.
<!-- ccbridge:end -->

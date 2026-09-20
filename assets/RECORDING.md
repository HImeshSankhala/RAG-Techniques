# Recording the three README demos

The README links three files from this directory. Until they exist the README still reads
correctly — it describes each demo in prose and says they may not be recorded yet — so
there is no rush and nothing is broken in the meantime.

| File | Shows |
|---|---|
| `assets/playground.gif` | One technique answering, with its passages and timing trace |
| `assets/compare-divergence.gif` | Standard vs Fusion diverging, then a 100%-overlap control |
| `assets/local-vs-haiku.gif` | The same query on the local model and on Haiku |

**Use exactly these filenames.** The README already points at them.

## Before you record

```bash
make index      # first: the eval harness exits 1 on an empty index
make eval
make dev
```

Four of the five compare presets carry a claim id (`compare.preset-*`) that pins their
caption to *this* index. Re-chunking the corpus can silently falsify one — it already did
once, which is why the harness exists. If those four come back `ok`, the queries below
still behave the way the captions say. **If any fail, stop and fix the caption, not the
recording.** Expected output ends with `25 claims: 25 ok, 0 failed`.

Two things `make eval` cannot check for you: the fifth preset (the two-part Dynamo/Raft
question) makes a claim about pass counts rather than retrieval, and no harness checks the
latency or cost figures. Watch those with your own eyes while filming.

None of the three demos films Graph RAG, so `make graph` is not needed here.

## Setup for every take

- Window 1280×800, browser zoom 100%.
- Hide bookmarks, quit anything that shows notifications, use a plain new profile.
- Let the model answer once before filming so the weights are resident — a cold
  `qwen3:8b` adds several seconds that are about your disk, not about RAG.
- Do not trim the wait. A local answer takes ~15s and Multi-Pass ~45s, and that latency is
  half of what these techniques are teaching. Cutting it makes the demo lie in the
  flattering direction.

---

## 1. `playground.gif`

1. Go to `http://localhost:3000/playground`.
2. **Technique** → `Standard RAG`. **Model** → `qwen3:8b`.
3. Click the preset chip **"How does Dynamo handle conflicting concurrent writes?"**
   (it is one of the four under the query box).
4. Press **Run** and let it finish — do not cut the wait.
5. Scroll down to the retrieved passages so their sources are legible.
6. Expand the steps trace and pause a beat on the per-stage timings.

Must be visible by the end: the answer, the passages it came from, and the trace. Stop
recording with the trace open.

## 2. `compare-divergence.gif` — the headline

This is the one that carries the project. It is two takes in one file.

**Take A — they diverge.**

1. Go to `http://localhost:3000/compare`.
2. Side **A** → `Standard RAG`. Side **B** → `Fusion RAG`. Both on `qwen3:8b`.
3. Click the preset **"What are reversed hostnames used for?"** — type it exactly if you
   enter it by hand, including the question mark.
4. **Run**, wait for both sides.
5. Hold on the diff summary row long enough to read it: the evidence overlap and the
   chunks only one side saw.

What it should show, measured at `top_k=4`: **1 of 4 overlap**. Dense retrieval returns
`chubby.md#2, gfs.md#3, chubby.md#4, cassandra.md#0` — no `bigtable.md` at all — while the
fused side pulls `bigtable.md#0` into the evidence. BM25 leads with it.

**Take B — the control, and do not skip it.**

6. Same two sides. Switch the query to the preset **"How does Raft elect a leader?"**.
7. **Run**, and hold on the diff row again.

This one is **4 of 4 overlap** — fusion returns exactly what Standard already had. Filming
only Take A would teach that fusion always helps, which is false and is the more expensive
mistake for a reader to make.

## 3. `local-vs-haiku.gif`

Costs about **$0.002** of real money. It is the only demo that does.

**This one needs a key first**, or the Haiku option is greyed out and step 2 is impossible
— the API reports the model unavailable and the `<option>` is disabled:

```bash
cp backend/.env.example backend/.env   # then set ANTHROPIC_API_KEY
```

Restart the backend so it picks the key up. Set a spend limit in the Anthropic Console and
keep auto-reload off before you do this.

1. `http://localhost:3000/compare`.
2. Side **A** → `Standard RAG` on `qwen3:8b`. Side **B** → `Standard RAG` on
   `claude-haiku-4-5`. Same technique on both sides — the technique is the control and the
   *model* is the variable.
3. Query: the preset **"What is hinted handoff?"**.
4. **Run**. Hold on the row where cost and latency differ.

**Caveat to keep in mind while filming:** the spend badge reads `backend/.usage.json`,
which is gitignored and per-machine. Whatever total it shows is *your* history, not
something a viewer can reproduce. Frame so the badge is incidental, or accept that the
number is illustrative — do not let it read as a published figure.

---

## Encoding

Record to `.mov`, then convert with a two-pass palette — a single-pass GIF from a screen
recording bands badly on text:

```bash
ffmpeg -i take.mov -vf "fps=12,scale=1280:-1:flags=lanczos,palettegen=stats_mode=diff" \
  -y /tmp/palette.png

ffmpeg -i take.mov -i /tmp/palette.png \
  -lavfi "fps=12,scale=1280:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" \
  -y assets/compare-divergence.gif
```

**Size: measure, then decide.** No budget is stated here because nobody has encoded one of
these yet, and a number invented in advance is the kind of claim this repo keeps having to
retract. Encode the first one, run `ls -lh assets/`, and set a ceiling you are willing to
carry in git forever — these files are permanent.

If one comes out too big, in this order:
1. drop to `fps=10`,
2. `scale=960:-1`, then `640:-1`,
3. tighten the framing.

Shorten the clip last. The waiting is the content.

## After recording

The README currently *describes* the three demos and names their paths. Once the files
exist, embed them — the markup is already sitting commented-out in README.md next to the
description of each demo, so it is an uncomment, not a rewrite.

A GIF is the one artifact in this repo that nothing tests and CI cannot regenerate. If the
UI changes, it is stale and silent about it. Re-run `make eval` and re-watch these before
trusting them again.

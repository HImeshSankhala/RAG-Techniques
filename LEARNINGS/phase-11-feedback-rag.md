# Phase 11 — Feedback-Based RAG

Phase 10 asked a human mid-run and threw the answer away when the request ended. This phase keeps it.
A thumbs up or down on a passage becomes a row in SQLite, and every later run of this technique reads
those rows back and reorders its candidates before answering.

    POST /api/run       {technique: "feedback-rag"} -> 12 candidates, reranked by stored votes, top 4 answered
    POST /api/feedback  {technique, query, chunk_ids, rating}  -> one row per passage

## The problem

An embedding model scores a passage against a question. It cannot know that *this* reader, or this
team, keeps finding a particular passage useless, or that "the pipeline" means the deploy tool and
not the CI one. That knowledge exists only in usage, and feedback is the cheapest way to capture it:
no retraining, no labelled set, one row per click.

## Why this design, against the naive alternatives

**A rank shift, not `similarity + α · votes`.** The obvious formula adds a weight to the cosine
score, and it is the Phase 4 mistake wearing different clothes. Measured on this corpus: the top-12
cosine band is **0.136-0.256** for "What is hinted handoff?" and **0.452-0.762** for "How does Dynamo
handle conflicting concurrent writes?". A single α that nudges the second query would flatten the
first. Ranks are the common currency, so a vote buys *places*:

    position = dense_rank − SHIFT · clamp(net_votes, −CAP, CAP)      # SHIFT = 2, CAP = 3
    order by (position, dense_rank)

Ties go to the retriever, so feedback never wins a coin flip against similarity.

**Over-fetch 12, answer from 4.** Reranking the final 4 could only permute what Standard RAG already
had. Fetching 12 gives feedback somewhere to promote from. The cost is an honest recall limit: a
passage below rank 12 is not a candidate and no number of votes can reach it. That is a limit of the
over-fetch, not of feedback, and the trace says so separately.

**Votes are keyed to the passage TEXT, not its id.** `chunk_id` is positional (`raft.md#1`) and
`make index` rebuilds it, so a stored vote keyed by id would silently apply to text nobody rated —
the same hazard Phase 10 hit with drafts. The key is `sha256(text)[:16]`, taken server-side from the
index at vote time. A passage that merely moves keeps its votes; a passage whose text changed loses
them. Both directions are tested.

**Append-only, no dedupe.** Every click is a row, so three clicks are net +3. This is what makes the
exposure problem real rather than theoretical, and the UI says "every click counts" instead of
pretending one reader equals one judgment.

**No `core/` changes and no shared db helper.** The reranker has one user, so it lives in the
pipeline. `feedback_rag` and `interactive_rag` are two SQLite users, not three, so the connection
pattern is repeated rather than extracted — the rule of three is not met yet.

**One new `Metadata` field: `feedback_votes`.** Every other technique is a function of its query.
This one is not, and the compare view had no way to say that: it would have reported accumulated
history as a retrieval disagreement. The field is 0 for the other six, and `DiffSummary` appends a
clause naming the count when a side used votes. That branch is the phase's only Vitest case, because
it is a pure function that decides what the reader is told.

## The algorithm, and its complexity

Per run: one dense query for 12 candidates (HNSW, ~O(log N) per lookup), one indexed SQLite read over
at most 12 hashes (`O(k' log V + matches)`), then `O(k' log k')` to sort — 12 items, microseconds.
Zero extra LLM calls: cost is identical to Standard RAG, one capped answer call.

**The sort key is not a rank.** A passage with key 7 can still finish 6th, because the passages it
overtook shift too. Every number in the trace ("rank 8 → 3", "pushed out of the top 4") is read off
the final sorted list. The reachability arithmetic is likewise only true for a *lone* vote: at
SHIFT=2 a lone +3 reaches the top 4 from as deep as rank 9, but with the ranks above it voted down,
even rank 12 gets in. There is a unit test for exactly that, because the isolated arithmetic is the
easy thing to teach wrongly.

## Failure mode: the bubble, measured on this corpus

Votes are global, so a *correct* judgment on one question silently reranks every other. Measured
live, on `qwen3:8b` with 43 chunks:

1. On "How are stragglers handled in MapReduce?", upvote `mapreduce.md#2` — the only passage in the
   corpus that mentions stragglers. Three clicks, net +3.
2. Ask "How does Raft elect a leader?", which shares no vocabulary with it. `mapreduce.md#2` was
   dense rank 8 there. It lands at **rank 3**, and `raft.md#1` — the passage headed "Leader
   election" — is pushed from rank 4 to rank 5, out of the answer.
3. The Raft reader now sees an irrelevant passage and downvotes it, which also demotes it for the
   MapReduce question. Two correct judgments, one number, fighting.

Across 8 preset queries × the 7 others, 5 pairs lost a passage containing their own answer term at
these settings. At one vote per judgment, none did — the harm needs repetition, which is exactly the
"boosted → seen more → rated more" loop the learn page describes.

Then the dead end: a demoted passage is no longer shown, so it has no thumbs. There is no way back
through the product, which is why the reset is `make reset-feedback` and not a button. The standard
fixes — decay, exposure normalisation, and above all an **exploration slot** that reserves one
top-k place for an unboosted candidate — are deliberately not built here, and the trace names them
on every run so the omission cannot be mistaken for completeness.

## Second failure mode: the vote that arrives after a re-index

The server hashes a passage's text when the vote arrives. If the index is rebuilt between a result
being rendered and its thumbs being clicked, the vote is hashed against whatever text now sits at
that id — a vote on text the reader never saw. Accepted rather than mitigated: the API contract
carries no text to check against, and the window is seconds on a local single-user app. Worth
knowing that "keyed by content" removes the *storage* hazard, not the *timing* one.

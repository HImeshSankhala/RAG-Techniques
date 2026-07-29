# Raft and Paxos

Consensus is the problem of getting a group of machines to agree on a value even
when some of them fail. It underpins replicated state machines, and therefore most
systems that claim to be both consistent and fault-tolerant.

## Paxos

Leslie Lamport introduced Paxos in a paper written in 1990 and published in 1998
as "The Part-Time Parliament," framed as an account of an ancient legislature.
The framing obscured the algorithm badly enough that Lamport published a plainer
restatement, "Paxos Made Simple," in 2001.

Paxos assigns three roles: proposers, acceptors, and learners. A proposer picks a
proposal number and asks acceptors to promise not to accept anything lower. If a
majority promises, the proposer sends a value, and if a majority accepts it, the
value is chosen. Majorities are what make this safe — any two majorities of the
same set share at least one member, so no two conflicting values can both be
chosen.

Single-decree Paxos agrees on one value. Real systems need a sequence of values,
which requires Multi-Paxos — and Multi-Paxos is where the difficulty concentrates.
The original papers describe the single-decree case precisely and leave the
practical extension largely to the reader, so implementations diverged and the
gap between the published algorithm and a working system stayed wide.

## Raft

Diego Ongaro and John Ousterhout published Raft in 2014 with an explicit design
goal that was unusual: understandability. The paper argues that Paxos's opacity
had real costs — it made systems harder to build correctly and harder to teach —
and that an algorithm equivalent in performance but easier to reason about would
be more useful in practice.

Raft decomposes consensus into three subproblems that can be understood
separately: leader election, log replication, and safety.

### Leader election

Raft time is divided into **terms**, each numbered, each beginning with an
election. Every server is a follower, a candidate, or a leader. A follower that
hears nothing from a leader before its election timeout expires becomes a
candidate, increments the term, and requests votes.

A candidate that receives votes from a majority becomes leader. Election timeouts
are randomized, which is what prevents repeated split votes: servers rarely time
out simultaneously, so one candidate usually starts first and wins.

Raft is a **strong leader** design. All client requests go through the leader, and
log entries flow only from leader to followers. Paxos allows any node to propose;
Raft's restriction removes a large class of interactions and is a major part of
why it is easier to follow.

### Log replication

The leader appends a client command to its log and sends it to followers. Once a
majority have stored the entry, the leader marks it **committed**, applies it to
its state machine, and returns the result.

The Log Matching Property keeps logs consistent: if two logs contain an entry with
the same index and term, then the logs are identical in all preceding entries.
The leader maintains this by including the index and term of the preceding entry
in every append; a follower whose log does not match rejects the append, and the
leader walks backward until it finds agreement.

### Safety

Raft restricts which servers may become leader: a candidate cannot win unless its
log is at least as up to date as a majority of the cluster's. This guarantees a
new leader already holds every committed entry, so committed entries are never
lost or overwritten.

## Trade-offs

Both algorithms tolerate the failure of a minority of servers — a cluster of five
survives two failures. Both require a majority to make progress, so a partitioned
minority stops accepting writes. This is the choice Dynamo declines to make: Raft
and Paxos sacrifice availability during partitions to keep replicas consistent,
while Dynamo sacrifices consistency to stay writable.

The strong leader is Raft's clearest trade-off. It simplifies reasoning but makes
the leader a throughput bottleneck and a single point of disruption — a failed
leader stalls the cluster for an election timeout.

Raft's understandability goal was borne out in adoption. It is the consensus layer
in etcd, Consul, and CockroachDB, and etcd's role in Kubernetes puts a Raft
implementation under a large share of modern infrastructure.

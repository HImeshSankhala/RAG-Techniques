# Chubby

Chubby is Google's distributed lock service, described in a 2006 OSDI paper by
Mike Burrows. Its stated purpose is not raw performance but making distributed
consensus available to ordinary engineers: most of its clients need to elect a
leader or agree on a small piece of configuration, and would otherwise each
implement Paxos badly.

## Locks as an interface to consensus

The paper argues a lock service is the right packaging. A library implementing
consensus would require every application to be restructured around it, while a
lock service can be dropped into an existing system. It also gives a place to
store the small amounts of metadata that leader election tends to need, and it
lets many clients share one well-tested consensus implementation.

Chubby offers **coarse-grained** locks — held for hours or days, such as "I am the
current master" — rather than fine-grained locks held for seconds. Coarse locks
mean lock traffic is low relative to client activity, and a Chubby outage is
survivable because clients hold their locks through it.

Locks are **advisory**: nothing prevents a client from touching data it has not
locked. Chubby's authors chose this because enforcement would require Chubby to
sit between clients and their data, which is exactly the bottleneck the design
avoids.

## Interface and structure

Chubby exposes a file system-like namespace of small files and directories, such
as `/ls/cell/app/master`. Any file can act as a lock, and a file can hold data —
capped at **256 KB**, because Chubby is a coordination service and not a store.

A typical Chubby **cell** has **five replicas**, of which one is the **master**.
Replicas elect the master using a **Paxos**-based replicated log, and the master
holds a **master lease** for a few seconds, renewed continuously. All client reads
and writes go through the master, so consistency needs no client-side protocol.

Clients keep a **session** with the cell, maintained by periodic **KeepAlive**
handshakes. If a session lapses without renewal the client's locks and handles are
released, which is how Chubby reclaims state from a dead client. Clients cache
file data aggressively and the master **invalidates** those caches on write, which
keeps the common read path off the network entirely.

Clients can subscribe to **events** — file contents changed, master failed over,
handle invalid — so a system waiting on a leader change learns about it without
polling.

A **sequencer** addresses a subtle failure: a client can acquire a lock, stall,
and issue a request after the lock has been given away. A sequencer is a token
describing the lock's state at acquisition time, which the client passes to the
service it is protecting, and that service rejects tokens that are out of date.

## Consensus underneath

The replicated log beneath Chubby is Paxos, which is what makes the service
correct under partitions and replica failure. Paxos is famously difficult to
reason about and to implement, and that difficulty is precisely the motivation
Diego Ongaro and John Ousterhout give for designing Raft as an understandable
alternative — Raft's decomposition into leader election, log replication, and
safety is a direct response to Paxos being hard to teach and hard to get right.

Chubby's implementers make the same point from experience: the published Paxos
descriptions left a large gap between algorithm and working system, and closing it
took substantial unpublished engineering.

## Who depends on it

Bigtable uses Chubby in several places at once: to store the pointer to its root
tablet, to elect and track its master, to discover tablet servers and detect their
failure, and to hold column-family schema information. A Bigtable cluster cannot
serve data while Chubby is unavailable, which makes Chubby's availability a hard
floor under Bigtable's.

GFS uses it for master election. Google's cluster naming service is layered on it
as well, which the paper notes was an unplanned but dominant use.

Apache ZooKeeper is the open-source system occupying the same niche, with a
comparable namespace-plus-watches interface over its own consensus protocol, Zab.

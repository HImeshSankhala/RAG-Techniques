# Spanner

Spanner is Google's globally distributed database, described in a 2012 OSDI paper
by James Corbett and colleagues. It is notable for providing **external
consistency** — the strongest guarantee in practice — across datacenters on
different continents, which the field had largely assumed was unaffordable.

## The guarantee

External consistency means that if a transaction T1 commits before T2 starts, then
T1's timestamp is smaller than T2's. Put plainly, the database behaves as if
transactions ran one at a time in the order they actually happened, even when they
executed on machines thousands of kilometres apart.

This is the opposite end of the spectrum from Dynamo. Dynamo accepts writes during
partitions and hands concurrent versions back to the application to reconcile;
Spanner refuses to let a transaction commit until it can prove its timestamp
ordering is safe. One buys availability with consistency, the other buys
consistency with latency.

## TrueTime

The mechanism is **TrueTime**, an API that exposes clock uncertainty instead of
hiding it. `TT.now()` does not return a timestamp — it returns an *interval*
`[earliest, latest]` that is guaranteed to contain the true absolute time.

TrueTime is backed by two independent hardware sources with uncorrelated failure
modes: **GPS receivers** and **atomic clocks**. GPS can fail through antenna
faults or jamming; atomic clocks drift slowly. Using both means a failure of one
kind does not silently corrupt the time base. Time masters in each datacenter
serve the interval, and the uncertainty **epsilon** is typically a few
milliseconds.

The trick is **commit wait**. A transaction picks a commit timestamp, then
deliberately waits until TrueTime says that timestamp is certainly in the past
before releasing its locks. The wait is on the order of epsilon — single-digit
milliseconds — and it is what converts a bounded clock uncertainty into a hard
ordering guarantee. Spanner does not need perfectly synchronised clocks; it needs
clocks that are honest about how wrong they might be.

Because timestamps are meaningful, a read-only transaction at a past timestamp
needs no locks at all, which makes consistent snapshot reads cheap.

## Structure

Data is organised into **directories** — contiguous key ranges that share a prefix
and are the unit of movement between machines. Directories live in **tablets**,
and each tablet is replicated across datacenters by its own **Paxos** group, with
a long-lived leader.

A transaction touching one Paxos group commits through that group alone. A
transaction spanning groups uses **two-phase commit** layered over Paxos, with one
group's leader acting as coordinator. Two-phase commit is normally criticised for
blocking when the coordinator dies; here each participant is itself a Paxos group,
so a failed coordinator is replaced by its own replicas rather than stalling the
transaction.

Spanner exposes a SQL-like query language and an explicit schema, and it supports
schema changes at a future timestamp so they do not block ongoing work.

## Lineage and trade-offs

Spanner sits at the end of the Google storage line. It is built on Colossus, the
successor to GFS, and it was explicitly motivated by Bigtable's limitations —
teams found Bigtable's lack of cross-row transactions and of a schema hard to
build on, and had been layering their own transaction systems over it.

It uses Paxos for replication, the same protocol Chubby uses for its replicated
log, and for the same reason: it is the part that has to be correct under
partition.

The costs are real. Writes pay a Paxos round plus commit wait, so single-write
latency is worse than an eventually consistent store. The system depends on
special hardware in every datacenter, which is why the design was not portable
until cloud vendors offered it as a service. And latency scales with clock
uncertainty — if epsilon grew to seconds, commit wait would make the system
unusable, so the engineering effort spent keeping epsilon small is not an
optimisation but a requirement.

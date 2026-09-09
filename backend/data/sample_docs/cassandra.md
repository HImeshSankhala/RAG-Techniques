# Cassandra

Apache Cassandra is a distributed store built at Facebook for Inbox Search and
described in a 2009 paper by Avinash Lakshman and Prashant Malik. It is the
clearest example of a system assembled from two earlier designs: it takes its
distribution and replication from Amazon's Dynamo, and its data and storage model
from Google's Bigtable.

## What came from Dynamo

The cluster is a ring with no master. Data is partitioned by **consistent
hashing**, with each node responsible for the range between it and its
predecessor, and modern versions assign each machine many **virtual nodes** so
load spreads evenly. Any node can serve any request, acting as a **coordinator**
that forwards to the replicas.

Replication is by **replication factor** rather than a fixed count, and placement
strategies can be made rack- and datacenter-aware so replicas do not share a
failure domain.

Consistency is **tunable per query**, which is Dynamo's R and W exposed directly
to the application: `ONE`, `QUORUM`, `LOCAL_QUORUM`, `ALL`. Choosing R + W greater
than the replication factor gives read-your-writes; choosing `ONE` for both gives
speed and no such guarantee. The same query can be issued at different consistency
levels by different parts of an application.

Failure handling is Dynamo's as well. Membership and liveness spread by
**gossip**, with an accrual failure detector that reports a suspicion level rather
than a boolean. **Hinted handoff** stores a write destined for an unreachable
replica on another node and delivers it later. **Read repair** compares replica
responses on the read path and pushes the newest value to whichever replica is
behind, and anti-entropy repair uses **Merkle trees** to reconcile whole ranges.

## What came from Bigtable

The storage engine is Bigtable's log-structured merge tree, essentially unchanged
in outline. A write is appended to a **commit log** for durability and applied to
an in-memory **memtable**. When the memtable is full it is flushed to an immutable
**SSTable** on disk, and background **compaction** merges SSTables, discarding
overwritten values and expired tombstones.

Reads merge the memtable with the relevant SSTables, and **Bloom filters** let a
read skip SSTables that certainly lack the key. The consequence is Bigtable's:
writes are fast because they touch memory and an append-only log, while reads may
have to consult several files.

The data model is also Bigtable's lineage — rows addressed by a partition key,
with **column families** grouping columns, and wide rows holding many columns.
Modern Cassandra presents this through CQL, a query language whose surface
resembles SQL but which deliberately omits joins, because a join would require
coordination the ring is designed to avoid.

## Where it differs from both

Cassandra does **not** use vector clocks. Dynamo returns concurrent versions to
the client and makes the application reconcile them; Cassandra instead resolves
conflicts with **last-write-wins** on a per-column timestamp. This is a real
trade, not a simplification: it removes the reconciliation burden Dynamo places on
applications, and it silently loses one of two concurrent updates, including when
clock skew between nodes makes "last" wrong.

**Tombstones** are the other consequence of a ring with no coordinator. A delete
cannot remove data immediately, because a replica that is down must not resurrect
the value when it returns, so a delete writes a marker that is retained for a
grace period before compaction discards it.

## Lineage

Cassandra's design is a deliberate merge: Dynamo's consistent hashing, virtual
nodes, tunable quorums, gossip, hinted handoff, and Merkle-tree anti-entropy, over
Bigtable's memtable, SSTables, compaction, Bloom filters, and column families.
Neither parent has both halves — Dynamo is a key-value store with no column model,
and Bigtable has a single master and no tunable consistency.

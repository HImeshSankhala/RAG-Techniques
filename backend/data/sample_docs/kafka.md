# Kafka

Apache Kafka is a distributed log, built at LinkedIn and described in a 2011 paper
by Jay Kreps, Neha Narkhede, and Jun Rao. It began as a solution to a pipeline
problem — many systems each wanting the same activity data — and its central move
is to treat the message queue as a durable, replayable log rather than a buffer
that empties as it is read.

## Topics, partitions, offsets

A **topic** is a named stream, divided into **partitions**. A partition is an
append-only sequence of records, and it is the unit of ordering, parallelism, and
replication. Ordering is guaranteed within a partition and not across a topic,
which is the trade that lets a topic scale horizontally.

Every record in a partition has a monotonically increasing **offset**. The offset
is the whole consumption model: a consumer's position is a number it controls, so
the broker keeps no per-message delivery state. Because of this, rewinding to
reprocess history is just seeking to an earlier offset, and a slow consumer costs
the broker nothing but disk.

This is the structural difference from a traditional queue. A queue deletes a
message once it is delivered, so it can serve one consumer group and cannot replay.
Kafka retains records for a configured **retention** period regardless of whether
anyone has read them, so several independent consumers read the same partition at
different positions.

**Consumer groups** provide the sharing rule: each partition is assigned to
exactly one consumer within a group, so adding consumers increases parallelism
only up to the partition count, while separate groups each receive everything.

## Storage

A partition is stored as a sequence of **segment** files. Writes append to the
active segment, and retention deletes whole segments rather than individual
records, which keeps cleanup cheap.

Kafka's throughput comes from refusing to be clever. It writes sequentially and
lets the operating system page cache do the caching rather than maintaining its
own in-process cache, and it sends data to the network with a zero-copy
`sendfile` path so records are not copied into user space at all. Producers and
consumers batch records, which amortises network and disk cost.

**Log compaction** is the alternative to time-based retention: instead of dropping
old records, it retains the most recent record for each key, so a compacted topic
converges to a snapshot of current state per key. This makes a topic usable as a
changelog for rebuilding a table.

## Replication and ISR

Each partition has a **leader** and a set of followers. Producers and consumers
talk only to the leader; followers replicate by fetching from it, exactly as a
consumer would.

The set of replicas that are sufficiently caught up is the **ISR**, the in-sync
replica set. A follower that falls too far behind is removed from the ISR, and
leader election chooses only from ISR members — which is what makes a committed
record survive failover.

Durability is a producer choice, made with **acks**. `acks=0` does not wait,
`acks=1` waits for the leader alone, and `acks=all` waits for the whole current
ISR. Pairing `acks=all` with `min.insync.replicas` is what turns "replicated" into
an actual guarantee, since an ISR that has shrunk to one member makes `acks=all`
mean nothing.

An **idempotent producer** deduplicates retries using a producer id and sequence
number, which removes the duplicates that retrying an unacknowledged send would
otherwise create.

## Coordination

Kafka originally kept cluster metadata — broker membership, partition leadership,
configuration — in Apache ZooKeeper, using it much as Bigtable uses Chubby: for
membership, failure detection, and election.

Newer versions replace ZooKeeper with **KRaft**, which moves that metadata into an
internal Kafka topic replicated by **Raft**. The motivation is partly operational,
removing a second distributed system from every deployment, and partly scaling,
since ZooKeeper-era metadata handling limited the number of partitions a cluster
could carry. The result is a system whose control plane is replicated by the same
kind of consensus log as its data plane.

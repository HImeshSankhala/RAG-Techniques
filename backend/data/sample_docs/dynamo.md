# Dynamo

Dynamo is a highly available key-value store built at Amazon and described in a
2007 SOSP paper by Giuseppe DeCandia and colleagues. It was designed for services
like the shopping cart, where a rejected write costs a sale — so the system
prioritizes write availability over strong consistency.

## The design target

Dynamo's requirements invert the usual database defaults. It offers only key-value
access: no queries across records, no joins, no relational schema. In exchange it
guarantees that writes are accepted even during network partitions and server
failures.

The system is designed around a service-level agreement expressed at the 99.9th
percentile rather than the mean. Amazon's argument is that averages hide the
customers with the largest carts and the most history — exactly the customers a
retailer least wants to fail.

## Consistent hashing

Dynamo partitions data across nodes with **consistent hashing**. Node positions
and key hashes are placed on the same fixed circular space, and a key belongs to
the first node encountered walking clockwise from the key's position.

The property that matters is what happens when membership changes. With a naive
`hash(key) mod N` scheme, changing N remaps nearly every key. With consistent
hashing, adding or removing a node only affects keys between that node and its
predecessor — on average K/N keys for K keys and N nodes.

Basic consistent hashing distributes load unevenly, because randomly placed nodes
do not carve the ring into equal arcs. Dynamo fixes this with **virtual nodes**:
each physical machine takes many positions on the ring instead of one. This
evens out the distribution and lets a more powerful machine carry proportionally
more of the ring.

## Replication and quorums

Each key is replicated to the N nodes that follow its position on the ring, called
its preference list. Reads and writes use a quorum protocol with two tunable
parameters: R, the number of replicas that must respond to a read, and W, the
number that must acknowledge a write.

Setting R + W > N gives read-your-writes behavior, because any read quorum and any
write quorum must overlap in at least one node. Lowering W increases write
availability at the cost of that guarantee. A common Dynamo configuration is
N=3, R=2, W=2.

To keep writes flowing during failures, Dynamo uses **hinted handoff**: if a
replica is unreachable, the write goes to another node with a hint recording where
it belongs. When the intended node returns, the hint is delivered and the
temporary copy is removed.

## Vector clocks and conflicts

Because writes are accepted during partitions, two clients can update the same key
concurrently and produce divergent versions. Dynamo tracks causality with **vector
clocks** — a list of (node, counter) pairs attached to each version.

Comparing two vector clocks reveals whether one causally precedes the other, or
whether the two are concurrent. If one precedes the other, the older is discarded
automatically. If they are concurrent, Dynamo cannot decide, so it returns both to
the client and pushes reconciliation into the application.

For the shopping cart, that resolution is a union of the two carts. The effect is
visible to users: a deleted item occasionally reappears, which Amazon accepted as
cheaper than refusing a write.

Anti-entropy uses Merkle trees to find divergence between replicas. Each replica
builds a hash tree over its key range, and two replicas compare roots — if the
roots match, the ranges are identical and nothing transfers. Only differing
subtrees are walked, so the data exchanged is proportional to the differences
rather than to the size of the dataset.

## Influence

Dynamo's ideas — consistent hashing with virtual nodes, tunable quorums, vector
clocks, anti-entropy with Merkle trees — became standard equipment. Apache
Cassandra pairs Dynamo's replication and partitioning with Bigtable's
column-family data model, and Riak implements the Dynamo design closely. Amazon's
later DynamoDB is a managed service that shares the name and much of the
philosophy, but is a distinct system.

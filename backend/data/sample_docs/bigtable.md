# Bigtable

Bigtable is a distributed storage system for structured data, built at Google and
described in a 2006 OSDI paper by Fay Chang and colleagues. It was designed for
workloads that MapReduce handles badly: low-latency reads and writes against
petabyte-scale datasets.

## The data model

Bigtable is not a relational database. It is a sparse, distributed, persistent,
multi-dimensional sorted map. A value is addressed by a row key, a column key, and
a timestamp.

Rows are kept in lexicographic order by row key, and this ordering is the main
tool an application has for controlling performance. Rows that sort near each
other are stored near each other, so a scan over a key range is efficient. The
canonical example is storing web pages under reversed hostnames — `com.google.maps`
rather than `maps.google.com` — so that all pages from one domain form a
contiguous range.

Columns are grouped into **column families**, which are the unit of access control
and the unit at which compression is configured. A family must be created before
data can be written to any column in it, but the number of columns within a family
is unbounded.

Each cell can hold multiple versions, indexed by timestamp. Bigtable can be
configured to garbage-collect old versions automatically, either keeping the last
N versions or discarding versions older than a cutoff.

## Tablets

A table is split by row range into **tablets**, and the tablet is the unit of
distribution and load balancing. A tablet is roughly 100–200 MB by default. As a
tablet grows it splits; as tables shrink, adjacent tablets merge.

A three-level hierarchy locates tablets, structured like a B+ tree. A file in
Chubby, Google's distributed lock service, points at the root tablet. The root
tablet holds the locations of all METADATA tablets, and those in turn hold the
locations of the user tablets. This structure addresses a very large number of
tablets in three lookups, and clients cache locations so most reads skip the
lookup entirely.

## Storage: SSTables and the memtable

Persistent data lives in SSTables — immutable, sorted files of key-value pairs,
stored in the Google File System. Immutability is a deliberate simplification: a
file that never changes needs no concurrency control for reads, and it can be
cached and shared freely.

Writes do not modify SSTables. An incoming write is appended to a commit log and
applied to an in-memory sorted buffer called the **memtable**. When the memtable
grows past a threshold it is frozen and written out as a new SSTable — a *minor
compaction*. Periodically, background *merging compactions* combine several
SSTables into one, discarding deleted and superseded entries.

A read must therefore consult a merged view of the memtable and the relevant
SSTables. Bloom filters let a read skip SSTables that certainly do not contain the
requested row, which removes most of the disk seeks a naive implementation would
perform.

This write path — sequential log, in-memory buffer, periodic merge — is the
log-structured merge tree, and it is the design that makes writes fast: a write
touches memory and an append-only log, never a random disk location.

## Dependencies and influence

Bigtable is built on several other Google systems: GFS stores its SSTables and
logs, Chubby holds its root metadata and handles master election, and its own
cluster management layer schedules tablet servers.

Its design propagated widely. Apache HBase is a direct open-source descendant, and
Apache Cassandra combines Bigtable's column-family data model with the replication
approach of Amazon's Dynamo.

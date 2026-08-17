# Google File System

The Google File System (GFS) is a distributed file system described in a 2003 SOSP
paper by Sanjay Ghemawat, Howard Gobioff, and Shun-Tak Leung. It was built for a
workload Google actually had rather than the one file systems traditionally
assume: huge files, mostly appended to rather than overwritten, read by large
streaming scans, on hardware that fails constantly.

## Assumptions that shaped it

The paper is unusually explicit that component failure is the normal case, not an
exception. With thousands of commodity machines, something is always broken, so
monitoring, fault tolerance, and automatic recovery are built into the design
rather than bolted on.

Files are huge — multi-gigabyte is typical — and small files are not optimized
for. The dominant write pattern is appending; random writes within a file are rare
enough that GFS does not try to make them fast. Reads are mostly large sequential
scans, which is what a MapReduce job does to its input.

## Architecture: one master, many chunkservers

A GFS cluster has a **single master** and many **chunkservers**. Files are divided
into fixed-size **chunks** of 64 MB, each identified by a globally unique 64-bit
**chunk handle**. Chunks are stored as ordinary Linux files on chunkservers and
replicated, by default to three chunkservers.

The 64 MB chunk size is large deliberately. It reduces client-master interaction,
lets a client hold a persistent TCP connection to a chunkserver for a long time,
and shrinks the metadata enough that the master can keep all of it in memory.

The master holds the file namespace, access control, the mapping from files to
chunks, and current chunk locations. It does **not** sit on the data path: a
client asks the master which chunkservers hold a chunk, caches that answer, and
then talks to chunkservers directly. Keeping bulk data off the master is what lets
a single master serve a large cluster without becoming the bottleneck.

Master state is kept durable through an **operation log** replicated to remote
machines, with periodic checkpoints so recovery does not replay everything. Chunk
locations are the exception — the master does not persist them. It asks
chunkservers at startup and keeps up to date through regular **heartbeat**
messages, because the chunkserver is the authority on which chunks it actually
has.

## Consistency and record append

GFS offers a deliberately relaxed consistency model. A file region is
*consistent* if all clients see the same data, and *defined* if it is consistent
and clients see a mutation in its entirety. Concurrent successful writes leave a
region consistent but undefined — readers see a mix of fragments from several
mutations.

Mutations are ordered by a **lease** the master grants to one replica, the
**primary**. The primary picks a serial order for all mutations to that chunk and
the secondaries follow it, so replicas stay consistent without the master
mediating every write.

The most-used operation is **record append**, which appends a record at an offset
GFS chooses and returns that offset. It guarantees the record is written
atomically at least once, but a failed attempt may leave padding or duplicates
behind. Applications are expected to cope using checksums and record identifiers,
which is much cheaper than making append exactly-once.

Stale replicas are detected with a **chunk version number**, incremented whenever
the master grants a new lease. A chunkserver that missed mutations while down
comes back with an old version number and is garbage-collected rather than served.

Deletion is lazy: a deleted file is renamed to a hidden name and removed some days
later, which makes accidental deletion recoverable.

## Dependencies and influence

GFS is the storage layer beneath much of the rest of Google's stack. MapReduce
reads its input from GFS and writes its output there, and the scheduler exploits
GFS chunk locations to place map tasks on machines that already hold a replica.
Bigtable stores its SSTables and its commit logs in GFS, which is why Bigtable's
own design can treat those files as durable and immutable.

Its successor, Colossus, removed the single-master limitation, and Spanner is
built on that later generation. The open-source HDFS follows the GFS design
closely, including the single-namenode architecture and the large default block
size.

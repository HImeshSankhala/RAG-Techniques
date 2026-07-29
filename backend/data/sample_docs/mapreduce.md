# MapReduce

MapReduce is a programming model for processing large datasets across clusters of
commodity machines. Jeffrey Dean and Sanjay Ghemawat described it in a 2004 paper
published at OSDI, based on systems already running inside Google.

## The programming model

A MapReduce job has two user-supplied functions. The **map** function takes an
input key-value pair and emits any number of intermediate key-value pairs. The
**reduce** function takes an intermediate key together with all values that were
emitted for that key, and produces the final output.

The canonical example is counting word occurrences across a large document
collection. The map function emits `(word, 1)` for every word it sees. The reduce
function receives a word and a list of counts, and sums them.

What makes the model useful is not the two functions — it is everything the
framework does between them. The runtime partitions the input, schedules work
across machines, groups intermediate pairs by key, handles machine failures, and
manages the network transfer. The programmer writes two functions that contain no
distributed systems code at all.

## The shuffle

The step between map and reduce is called the shuffle, and it is where most of the
cost lives. Every mapper's output must reach the reducer responsible for that key,
which means data moves across the network in an all-to-all pattern.

The framework assigns keys to reducers with a partitioning function, by default a
hash of the key modulo the number of reducers. Because all values for a given key
land on the same reducer, the reduce function can assume it sees the complete set.

## Fault tolerance

MapReduce assumes machines fail routinely, because on a cluster of thousands of
commodity machines they do. The master pings each worker periodically. When a
worker stops responding, any map task it completed is re-executed elsewhere —
completed map output lives on the failed machine's local disk and is no longer
reachable. Completed reduce tasks do not need re-execution, because their output
is written to a distributed file system.

This recovery strategy works because map and reduce functions are required to be
deterministic: re-running a task produces the same output, so a re-execution is
indistinguishable from the original.

## Stragglers

A single slow machine can delay an entire job, because the job is not finished
until its last task is. MapReduce handles these stragglers with backup execution:
near the end of a job, the master schedules duplicate copies of the remaining
in-progress tasks, and takes whichever finishes first. In the original paper this
technique cut one sort benchmark's completion time by 44 percent.

## Influence and limits

MapReduce shaped a decade of data infrastructure. Apache Hadoop was built as an
open-source implementation of the model, and the paper's ideas about scheduling
and fault tolerance appear throughout later systems.

Its limits are equally instructive. Every job writes its output to disk, so
multi-stage computations pay repeated serialization costs — the observation that
motivated later in-memory systems such as Apache Spark. The model also suits batch
processing rather than interactive queries: a MapReduce job's latency is measured
in minutes, which is why Google built separate systems, including Bigtable, for
low-latency access to structured data.

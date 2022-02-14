If a sequence has a length that is not a power of 2, virtual padding has to be inserted.

If we consider the padding values to be inserted in either the back or the front of the sequence, we can disable all of the threads that try to access them. This is the case because if we consider the padding values to be equal to smallest/largest value, all compares against them will fail and no swaps will be performed.

When sorting in descending order, -inf can be inserted in the back of the sequence without affecting its bitonic property. This is effective, since disabling the threads that access these padding values is the same as launching less threads.

When sorting in ascending order, -inf can be inserted in the front of the sequence without affecting its bitonic property. In this case, we can assign each thread a virtual id that it will use instead of its real one to decide which elements it will access. Then, these elements' addresses have to be adjusted since the padding elements are virtual and not actually present in memory. This is less efficient then in the descending case, since due to the indexing being based on a virtual thread id instead of a physical one, some groups of threads will access data in differing cache lines where it was previously in the same cache line. This could be solved by indexing the data back to front instead of using a thread id, but this is more complicated.

However, if we could have a bitonic sequence that is made up of a descending sequence and an ascending one instead of first an ascending one and then a descending one, the virtual thread id scheme wouldn't be needed at all.

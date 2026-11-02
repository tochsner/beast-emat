---
name: beast-unit-tests
description: Pitfalls when writing or debugging seeded BEAST unit tests of this repo, above all how BEAST's static Randomizer behaves with several threads and why Randomizer.setSeed does not make a fixture reproducible by itself. Use when writing statistical tests of operators (chains compared to an exact posterior or a reference sampler), when a seeded test is flaky or differs between runs, or when code under test starts threads.
---

# Seeded BEAST unit tests

## How `beast.base.util.Randomizer` works with threads

Read from the bytecode of beast-base 2.8.0-beta2 (`javap -p -c beast.base.util.Randomizer`). Check it again after an upgrade of beast-base.

- All draws go through a private `random()` that picks the generator **by thread name** (`Thread.currentThread().getName()`), not by thread identity. The name is never null, so the static main generator is only the seed source. Even the first draw of the test thread creates its own generator in `randomMap`.
- A generator that does not exist yet is created lazily on the first draw of a thread name. It is seeded with `<seed of the static generator> + randomMap.size()`. Its seed therefore depends on how many thread names have drawn before it, which depends on the order in which tests and threads ran earlier in the same JVM.
- `Randomizer.setSeed(s)` reseeds the static generator and **only the generators that exist at that moment**. The i-th entry of `randomMap.values()` gets `s + i`, in hash order of the thread names. Threads that draw for the first time after `setSeed` get the lazy seed above.
- Threads with the same name share one `MersenneTwisterFast`, which is not thread-safe. Thread pools that reuse names (for example `pool-1-thread-1` of several executors) interleave their draws.
- `randomMap` is a plain `HashMap` and the lazy creation is not synchronised. Several threads that draw for the first time at once can corrupt it. Only `setSeed` synchronises.

Consequences for tests:

1. A seeded test is only reproducible if all draws come from one thread. Code under test that starts threads (parallel likelihood, executors, parallel streams) makes the draws depend on scheduling and on earlier tests.
2. Call `Randomizer.setSeed` directly before the code whose draws matter, in the thread that runs it, not once in a static initialiser or a base class.
3. Do not compare two runs of a multi-threaded computation draw by draw. Compare statistics with a tolerance that covers the Monte Carlo error.
4. Surefire in this repo runs the tests of one module sequentially in one JVM (no parallel settings in `pom.xml`, no `junit-platform.properties`), so generators persist across test classes. Reseeding at the start of each test is what keeps tests independent of the test order.

## Do not rebuild a fixture and expect the same state

A seed does not guarantee that a fixture is rebuilt identically. In this repo, six-tip chains that were built with the same `setSeed(7)` before `ParsimonyMutationsInitialiser` still got different histories. The cause was not diagnosed. Two chains that sample different histories sample different posteriors, so a comparison between them looks like a bias of the sampler.

- Build the fixture **once**, snapshot its heights and mutations (`tree.getNodesAsArray()` heights and `mutations.getMutations(node)` per node), and restore the snapshot with `mutations.applyMutations(node, list, operator)` before each run.
- When two samplers disagree, first check that they started from the same state, then check against a third, trivial sampler (for example a random walk on one node at a time) before blaming the operator.

## Statistical tests of operators

- Compare an operator against an exact distribution or against an independent reference sampler on the same fixed state. Fix everything the operator does not move (here the histories), so that the target is one fixed density.
- Noise is not the same for all samplers. Two chains of a fast-mixing sampler can agree to 1e-3 or better, so a tolerance of 1e-2 is not generous. Run two seeds of the reference to see how large the noise really is before choosing a tolerance.
- Add a negative control: break the Hastings ratio on purpose and check that the test fails.
- The test loop must emulate BEAST: call `operator.setOperatorSchedule(new OperatorSchedule())` first, then `operator.accept()` after an accepted proposal and `operator.reject(0)` after a rejected one (`reject(-2)` if the proposal returned `Double.NEGATIVE_INFINITY`). Without the schedule, `accept()` throws a `NullPointerException`. Operators with internal state (a direction, a momentum) update it in these two methods.
- An operator that moves only a collective coordinate (a shared shift, a rotation along the sum of the scores) leaves the target invariant but is not ergodic on its own. A chain of only that operator looks biased against the exact answer, and the bias is the initial state, not an error of the Hastings ratio. Mix it with an operator that moves the nodes individually, and give the operator under test most of the proposals (here four in five) so that a wrong Hastings ratio is not diluted.
- Calibrate the tolerance on the noise of the reference, and check it against a deliberately wrong operator: with half of the proposals from an exact operator, a Hastings term that was removed completely still moved the means by only 0.01 to 0.02, which a tolerance of 0.02 does not see.

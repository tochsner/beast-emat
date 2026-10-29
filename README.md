## EMATs for BEAST 2

This project adds an implementation of Explicitly Annotated Mutation Trees (EMATs) into the BEAST 2 framework. It was heavily inspired by Delphy and the BEAST package [emat](https://github.com/BEAST2-Dev/emat).

## Preliminary Convergence Time

Running time until an ESS of at least 200 is reached on all parameters, for datasets of increasing size.


| Dataset                   | Sequences | Nucleotide diversity (π) | Missing fraction | Runtime   |
| ------------------------- | ---------: | ------------------------: | ----------------: | ---------: |
| Zika                      | 174       | 0.0036                   | 0.1778           | 12min     |
| RSV A                     | 215       | 0.0127                   | 0.0303           | 27min     |
| RSV B                     | 390       | 0.0082                   | 0.0323           | 31min     |
| Adenovirus B1             | 735       | 0.0684                   | 0.0799           | 17h 49min |
| HMPV                      | 986       | 0.1135                   | 0.3275           | \~19d     |
| Ebola                     | 1610      | 0.0009                   | 0.0162           | 1h 16min  |
| Influenza A (H1N1)        | 3746      | 0.0185                   | 0.0006           | 7h 38min  |
| Influenza A (H3N2)        | 4036      | 0.0115                   | 0.0186           | 6h 36min  |
| SARS-CoV-2 (2020, COG-UK) | 757       | 0.0002                   | 0.0181           | 13min     |
| SARS-CoV-2 (2020, COG-UK) | 5000      | 0.0003                   | 0.0134           | 6h 31min  |
| SARS-CoV-2 (2020, COG-UK) | 10000     | 0.0003                   | 0.0135           | 28h 23min |
| SARS-CoV-2 (2024 - 2026)  | 5800      | 0.0012                   | 0.0480           | \~5d      |


## Package Overview


| Package                       | Description                                                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `emat.state`                  | `Mutations` and `Mutation`: the mutations annotated on the branches of the tree.                                          |
| `emat.prior`                  | `GeneticPrior`: the genetic prior of the mutations (evolutionary model).                                                  |
| `emat.operators.topology`     | Subtree moves that change the tree topology; `mdspr` holds the buffer of mutation-directed SPR.                           |
| `emat.operators.nodetimes`    | Moves that change node heights; `batch` holds the batch displacement and its learned tree prior, `warp` the learned warp. |
| `emat.operators.mutations`    | Moves that only change the mutation histories on the branches.                                                            |
| `emat.stochasticmapping`      | Stochastic mapping of site histories (Jukes-Cantor, uniformised), and the samplers built on it.                           |
| `emat.helper`                 | Shared helpers such as Fitch parsimony, mutation paths and site changes.                                                  |
| `emat.initalisation`          | `ParsimonyMutationsInitialiser`: initialises the mutations from a parsimony reconstruction.                               |
| `emat.alignment`, `emat.tree` | Alignment filtering and tree interval utilities.                                                                          |
| `emat.benchmark`              | `TimedMCMC`: MCMC with timing for benchmarks.                                                                             |


## Operators

All operators keep the mutations on the tree consistent, and are grouped by what they change.

**Topology (`emat.operators.topology`)**


| Operator                      | Description                                                                                                                                                                   |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SubtreePruneRegraftOperator` | Abstract base of the subtree moves below; resamples the history on the affected branches.                                                                                     |
| `MutationDirectedSprOperator` | Regrafts a subtree preferably where the sequences are similar, using the mutations as a parsimony map. The subtree is picked uniformly or in proportion to its branch length. |


**Node times (`emat.operators.nodetimes`)**


| Operator                        | Description                                                                                                                                                                                                          |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `InnerNodeDisplacementOperator` | Moves an internal node within the range allowed by the mutations on its adjacent branches.                                                                                                                           |
| `BatchNodeDisplacementOperator` | Displaces a batch of internal nodes at once, using a learned approximation of the tree prior.                                                                                                                        |
| `RootScaleOperator`             | Scales the shorter root branch, which moves the root height.                                                                                                                                                         |
| `RatioScaleOperator`            | Scales the height ratio of a random internal node to its parent, both above its oldest tip, and rescales the subtree below so that all other ratios stay the same. Tunes a separate step for every clade size class. |
| `MutationRatioScaleOperator`    | Like `RatioScaleOperator`, but tunes a separate step for every mutation count class of the subtree, starting from the width c / √(1 + m) that m dragged mutations imply.                                             |
| `IntervalScaleOperator`         | Scales all intervals between nodes and their older children, keeping the tree valid for serial samples.                                                                                                              |
| `LearnedWarpOperator`           | Jointly moves the tree prior parameters, all internal node heights and the clock rate, using a time change learned from the chain.                                                                                   |


**Mutations (`emat.operators.mutations`)**


| Operator                    | Description                                                               |
| --------------------------- | ------------------------------------------------------------------------- |
| `MutationTimeOperator`      | Moves a single mutation between its neighbouring events at the same site. |
| `BranchReformOperator`      | Resamples the mutation history of a branch without changing the tree.     |
| `SiteHistoryGibbsOperator`  | Resamples the complete history of a site by stochastic mapping.           |
| `MissingCladeGibbsOperator` | Resamples the history of the clades whose data is missing around a tip.   |

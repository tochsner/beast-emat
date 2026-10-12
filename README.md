## EMATs for BEAST 2

This project adds an implementation of Explicitly Annotated Mutation Trees (EMATs) into the BEAST 2 framework. It was heavily inspired by Delphy.

## Current Results

Currently, we achieve the following rough runtimes:

| Type         | Num. Sequences | Hours to Converge |
|--------------|----------------|-------------------|
| SARS-CoV 2   | 787            | ~1.5              |
| SARS-CoV 2   | 5000           | ~10               |
| SARS-CoV 2   | 10000          | ~30               |
| Ebola        | 1610           | ~2                |
| Zika         | 174            | ~0.5              |

## Package Overview

| Package                         | Description                                                                                    |
|---------------------------------|------------------------------------------------------------------------------------------------|
| `emat.state`                    | `Mutations` and `Mutation`: the mutations annotated on the branches of the tree.               |
| `emat.prior`                    | `GeneticPrior`: the genetic prior of the mutations (evolutionary model).                       |
| `emat.operators`                | MCMC operators on the tree and its mutations (see below).                                      |
| `emat.stochasticmapping`        | Exact and approximate (Jukes-Cantor, uniformised) stochastic mapping of site histories.        |
| `emat.helper`                   | Helpers such as Fitch parsimony, site histories, and tree prior approximations.                |
| `emat.initalisation`            | `ParsimonyMutationsInitialiser`: initialises the mutations from a parsimony reconstruction.    |
| `emat.alignment`, `emat.tree`   | Alignment filtering and tree interval utilities.                                               |
| `emat.benchmark`                | `TimedMCMC`: MCMC with timing for benchmarks.                                                  |

## Operators

All operators live in `emat.operators` and keep the mutations on the tree consistent.

**Tree and height moves**

| Operator                       | Description                                                                                                 |
|--------------------------------|-------------------------------------------------------------------------------------------------------------|
| `InnerNodeDisplacementOperator`| Moves an internal node within the range allowed by the mutations on its adjacent branches.                 |
| `BatchNodeDisplacementOperator`| Displaces a batch of internal nodes at once, using a learned approximation of the tree prior.               |
| `InnerNodeResampleOperator`    | Moves an internal node between its older child and parent and resamples the three adjacent histories.      |
| `RootScaleOperator`            | Scales the shorter root branch, which moves the root height.                                                |
| `IntervalScaleOperator`        | Scales all intervals between nodes and their older children, keeping the tree valid for serial samples.    |
| `SubtreePruneRegraftOperator`  | Abstract base of the subtree moves below; resamples the history on the affected branches.                  |
| `SubtreeSlideOperator`         | Slides the parent of a subtree up or down the tree.                                                         |
| `WilsonBaldingOperator`        | Regrafts a subtree onto a uniformly chosen branch at a uniform height.                                      |
| `MutationDirectedSprOperator`  | Regrafts a subtree preferably where the sequences are similar, using the mutations as a parsimony map.      |

**Mutation and history moves**

| Operator                  | Description                                                                                           |
|---------------------------|-------------------------------------------------------------------------------------------------------|
| `MutationTimeOperator`    | Moves a single mutation between its neighbouring events at the same site.                             |
| `BranchReformOperator`    | Redraws the mutation history of a branch without changing the tree.                                   |
| `SiteHistoryGibbsOperator`| Resamples the complete history of a site by stochastic mapping.                                       |
| `SiteRegionGibbsOperator` | Resamples the history of a site on a connected region of the tree around an inner node.               |

**Wrappers**

| Operator                     | Description                                                                                         |
|------------------------------|-----------------------------------------------------------------------------------------------------|
| `ParameterFollowingOperator` | Wraps a tree operator and shifts tree prior parameters along with the change of the tree length.    |

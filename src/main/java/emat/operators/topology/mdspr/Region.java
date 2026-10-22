package emat.operators.topology.mdspr;

import beast.base.evolution.tree.Node;

/**
 * A part of a branch of the pruned tree with a constant sequence: the node below the
 * branch, the index of the part counted from the top of the branch, the height range
 * of the part that is older than X, and the number of sites where its sequence differs
 * from the sequence of X, leaving out the sites where the data of X is missing.
 */
public record Region(Node node, int index, double lowerHeight, double upperHeight, int numDifferences) {
}

package emat.operators.mdspr;

/**
 * The distance between the old and the new attachment point of a proposed move in the
 * pruned tree: the number of nodes and the length of the path between them.
 */
public record ProposedMove(int numNodes, double pathLength) {
}

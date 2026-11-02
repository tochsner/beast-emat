package emat.operators.nodetimes.height;

import beast.base.evolution.tree.Node;
import beast.base.inference.Operator;
import emat.operators.nodetimes.BranchTimeMapper;
import emat.state.Mutations;

/** Moves the height of an internal node together with the mutations on the three branches around it. */
public final class NodeHeightMover {

    private NodeHeightMover() {
    }

    /**
     * Sets the height of the given non-root node and maps the mutations on the branch above it and on the branches
     * of its children linearly onto the new branches, so that each mutation keeps its relative position. Returns the
     * log Jacobian of the map, which is the log ratio of the new to the old branch length once per mutation.
     */
    public static double moveNode(Mutations mutations, Node node, double newHeight, Operator operator) {
        double oldHeight = node.getHeight();
        double parentHeight = node.getParent().getHeight();

        node.setHeight(newHeight);

        double logJacobian = BranchTimeMapper.mapBranch(mutations, node, parentHeight, oldHeight, parentHeight, newHeight, operator);
        for (Node child : node.getChildren()) {
            logJacobian += BranchTimeMapper.mapBranch(mutations, child, oldHeight, child.getHeight(), newHeight, child.getHeight(), operator);
        }

        return logJacobian;
    }

    /** Returns the older of the two children of the given internal node. */
    public static Node getOlderChild(Node node) {
        return node.getLeft().getHeight() >= node.getRight().getHeight() ? node.getLeft() : node.getRight();
    }

    /** Returns the younger of the two children of the given internal node. */
    public static Node getYoungerChild(Node node) {
        return node.getLeft().getHeight() >= node.getRight().getHeight() ? node.getRight() : node.getLeft();
    }

}

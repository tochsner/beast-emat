package emat.operators.nodetimes;

import beast.base.evolution.tree.Node;
import beast.base.inference.Operator;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.List;

/** Maps the mutations of branches linearly along with their ends when the heights of nodes change. */
public final class BranchTimeMapper {

    private BranchTimeMapper() {
    }

    /**
     * Maps the mutations of every non-root branch from the old to the new heights of its ends, where the old heights
     * are given by node number. Returns the sum over the mutations of the log ratio of the new to the old branch
     * length, which is the log Jacobian of the map.
     */
    public static double mapMutations(Mutations mutations, Node[] nodes, double[] oldHeights, Operator operator) {
        double logJacobian = 0.0;

        for (Node node : nodes) {
            // the times of the root mutations are meaningless
            if (!node.isRoot()) {
                logJacobian += mapBranch(
                        mutations, node, oldHeights[node.getParent().getNr()], oldHeights[node.getNr()],
                        node.getParent().getHeight(), node.getHeight(), operator
                );
            }
        }

        return logJacobian;
    }

    /**
     * Maps the mutations on the branch above the given node linearly from the old to the new ends of the branch.
     * Returns the log Jacobian of the map, which is the log of the ratio of the new to the old branch length once
     * per mutation.
     */
    public static double mapBranch(Mutations mutations, Node node, double oldStartHeight, double oldEndHeight,
                                   double newStartHeight, double newEndHeight, Operator operator) {
        List<Mutation> branchMutations = mutations.getMutations(node);
        if (branchMutations.isEmpty()) {
            return 0.0;
        }

        List<Mutation> newBranchMutations = new ArrayList<>(branchMutations.size());
        for (Mutation mutation : branchMutations) {
            newBranchMutations.add(new Mutation(
                    mutation.nodeNr(),
                    mapHeight(mutation.time(), oldStartHeight, oldEndHeight, newStartHeight, newEndHeight),
                    mapHeight(mutation.timeOfPreviousMutation(), oldStartHeight, oldEndHeight, newStartHeight, newEndHeight),
                    mutation.site(), mutation.oldState(), mutation.newState()
            ));
        }

        mutations.applyMutations(node, newBranchMutations, operator);

        double lengthRatio = (newStartHeight - newEndHeight) / (oldStartHeight - oldEndHeight);
        return branchMutations.size() * Math.log(lengthRatio);
    }

    /**
     * Maps a height on a branch linearly from its old to its new ends. The start maps exactly
     * onto the new start, which keeps the times of previous mutations at the branch start
     * consistent, and rounding cannot push a height off the branch.
     */
    private static double mapHeight(double height, double oldStartHeight, double oldEndHeight, double newStartHeight, double newEndHeight) {
        if (height == oldStartHeight) {
            return newStartHeight;
        }

        double fraction = (height - oldEndHeight) / (oldStartHeight - oldEndHeight);
        double newHeight = newEndHeight + fraction * (newStartHeight - newEndHeight);
        return Math.min(Math.max(newHeight, newEndHeight), newStartHeight);
    }

}

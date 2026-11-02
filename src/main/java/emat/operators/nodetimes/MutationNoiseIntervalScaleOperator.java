package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import emat.diagnostics.DiagnosedOperator;
import emat.diagnostics.StepAnnotations;

import java.util.ArrayList;
import java.util.List;

@Description("Multiplies the interval between every internal node and its older child by exp(a), where the log factor a " +
        "of a node is the sum of a shared normal draw and an own normal draw whose standard deviation shrinks with the " +
        "number of mutations m on the two child branches of the node, as 1 / sqrt(1 + m). The genetic prior pins a branch " +
        "with many mutations to a length with relative width 1 / sqrt(1 + m), so nodes without mutations take the largest " +
        "steps. The shared draw moves the whole tree length. The mutation counts do not change with the move, and the older " +
        "child of a node is the higher of its new children. The mutations are mapped linearly within their branches.")
public class MutationNoiseIntervalScaleOperator extends ScaleTunedOperator implements DiagnosedOperator {

    final public Input<Double> scaleUpperLimit = new Input<>("upper", "upper limit of the scale factor", 10.0);
    final public Input<Double> scaleLowerLimit = new Input<>("lower", "lower limit of the scale factor", 1e-8);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "initial standard deviation of the shared draw, which scales the standard deviation of the own draws too", 0.03);
    final public Input<Double> sharedWeightInput = new Input<>("sharedWeight", "the weight of the shared draw relative to the own draws of nodes without mutations", 1.0);

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.scaleFactor = this.scaleFactorInput.get();
        this.upper = this.scaleUpperLimit.get();
        this.lower = this.scaleLowerLimit.get();
    }

    /**
     * Rebuilds the heights from the youngest nodes up, with every interval multiplied by exp(a_u) and a_u = s (w0 z0 + z_u / sqrt(1 + m_u)),
     * for independent standard normal z0 and z_u. The intervals are coordinates of the tree for the fixed topology and tip
     * heights, so the move is a bijection with the reverse log factors -a_u. The distribution of the log factors is
     * symmetric, as the mutation counts do not change, so the log Jacobian, the sum of the log factors, is the only term
     * besides the Jacobian of the mutation times.
     */
    @Override
    public double proposal() {
        Node[] nodes = this.tree.getNodesAsArray();
        double[] oldHeights = new double[nodes.length];
        for (Node node : nodes) {
            oldHeights[node.getNr()] = node.getHeight();
        }

        Node[] internalNodes = this.collectInternalNodesFromRoot();

        double sharedDraw = this.sharedWeightInput.get() * Randomizer.nextGaussian();
        double logFactorSum = 0.0;

        double[] newHeights = oldHeights.clone();
        for (int i = internalNodes.length - 1; i >= 0; i--) {
            Node node = internalNodes[i];
            double oldInterval = oldHeights[node.getNr()] - Math.max(oldHeights[node.getLeft().getNr()], oldHeights[node.getRight().getNr()]);

            int numMutations = this.mutations.getMutations(node.getLeft()).size() + this.mutations.getMutations(node.getRight()).size();
            double logFactor = this.scaleFactor * (sharedDraw + Randomizer.nextGaussian() / Math.sqrt(1.0 + numMutations));
            logFactorSum += logFactor;

            newHeights[node.getNr()] = Math.max(newHeights[node.getLeft().getNr()], newHeights[node.getRight().getNr()])
                    + oldInterval * Math.exp(logFactor);
        }
        for (Node node : internalNodes) {
            node.setHeight(newHeights[node.getNr()]);
        }

        StepAnnotations.set(0, logFactorSum / internalNodes.length);
        StepAnnotations.set(1, this.scaleFactor);

        return logFactorSum + BranchTimeMapper.mapMutations(this.mutations, nodes, oldHeights, this);
    }

    /** Returns the internal nodes with every node placed after its parent, starting with the root. */
    private Node[] collectInternalNodesFromRoot() {
        List<Node> internalNodes = new ArrayList<>();
        List<Node> stack = new ArrayList<>();
        stack.add(this.tree.getRoot());

        while (!stack.isEmpty()) {
            Node node = stack.removeLast();
            if (node.isLeaf()) {
                continue;
            }
            internalNodes.add(node);
            stack.add(node.getLeft());
            stack.add(node.getRight());
        }

        return internalNodes.toArray(new Node[0]);
    }

    @Override
    public String[] getAnnotationNames() {
        return new String[]{"meanLogFactor", "width"};
    }

}

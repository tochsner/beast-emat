package emat.adapters;

import adapters.Adapter;
import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;

import java.util.Arrays;
import java.util.List;

@Description("Immutable adapter that summarises a tree by the logarithm of its coalescent exposure, the sum over the " +
        "intervals between consecutive tips and coalescences of the number of pairs of lineages times the length of the interval.")
public class CoalescentExposureAdapter extends BEASTObject implements Adapter {

    public final Input<Tree> treeInput = new Input<>("tree", "The tree to summarise.", Input.Validate.REQUIRED);

    private Tree tree;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
    }

    @Override
    public int getNumImmutable() {
        return 1;
    }

    @Override
    public int getNumMutable() {
        return 0;
    }

    @Override
    public double[] getImmutable(int nodeId) {
        return new double[]{Math.log(this.computeCoalescentExposure())};
    }

    /**
     * Walks back in time from the youngest tip. A tip adds a lineage and a coalescence removes one, and a tip
     * is processed before a coalescence at the same height. The k lineages of an interval have k (k - 1) / 2 pairs
     * that can coalesce in it.
     */
    private double computeCoalescentExposure() {
        Node[] nodes = this.tree.getNodesAsArray();
        double[] tipHeights = new double[this.tree.getLeafNodeCount()];
        double[] coalescenceHeights = new double[this.tree.getInternalNodeCount()];

        int numTips = 0;
        int numCoalescences = 0;
        for (Node node : nodes) {
            if (node.isLeaf()) {
                tipHeights[numTips++] = node.getHeight();
            } else {
                coalescenceHeights[numCoalescences++] = node.getHeight();
            }
        }
        Arrays.sort(tipHeights);
        Arrays.sort(coalescenceHeights);

        double exposure = 0.0;
        int numLineages = 0;
        double previousHeight = Math.min(tipHeights[0], coalescenceHeights[0]);

        int tipIndex = 0;
        int coalescenceIndex = 0;
        while (tipIndex < numTips || coalescenceIndex < numCoalescences) {
            boolean isTip = coalescenceIndex == numCoalescences
                    || (tipIndex < numTips && tipHeights[tipIndex] <= coalescenceHeights[coalescenceIndex]);
            double height = isTip ? tipHeights[tipIndex] : coalescenceHeights[coalescenceIndex];

            exposure += 0.5 * numLineages * (numLineages - 1.0) * (height - previousHeight);
            previousHeight = height;

            if (isTip) {
                tipIndex++;
                numLineages++;
            } else {
                coalescenceIndex++;
                numLineages--;
            }
        }

        return exposure;
    }

    @Override
    public double[] getMutable(int nodeId) {
        return new double[0];
    }

    @Override
    public double update(double[] mutable, int nodeId) {
        return 0.0;
    }

    @Override
    public double getLogJacobianCorrection(int nodeId) {
        return 0.0;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree);
    }

}

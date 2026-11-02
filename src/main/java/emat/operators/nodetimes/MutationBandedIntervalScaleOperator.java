package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;

import java.util.Arrays;

@Description("Scales the intervals between the nodes of one mutation band and their older children by a common factor, " +
        "like the banded interval scaler, but the bands are formed by the number of mutations on the two child branches of " +
        "a node. The mutations stay on their branches when the heights move, so the bands do not change with the move. " +
        "Branches without mutations are only constrained by the tree prior and can take larger steps than branches that " +
        "many mutations pin down, and every band tunes its own scale factor.")
public class MutationBandedIntervalScaleOperator extends BandedIntervalScaleOperator {

    final public Input<String> mutationBoundsInput = new Input<>("mutationBounds",
            "the largest mutation counts of all bands but the last, separated by spaces: a node is in the first band whose bound is at least its mutation count", "0 2 6");

    private int[] bounds;

    @Override
    public void initAndValidate() {
        this.bounds = Arrays.stream(this.mutationBoundsInput.get().trim().split("\\s+")).mapToInt(Integer::parseInt).toArray();
        this.numBandsInput.setValue(this.bounds.length + 1, this);
        super.initAndValidate();
    }

    /** Returns the band of every internal node by its index, from the band of the fewest mutations to the band of the most. */
    @Override
    protected int[] assignBands(Node[] internalNodes) {
        int[] bandOfNode = new int[internalNodes.length];
        for (int i = 0; i < internalNodes.length; i++) {
            Node node = internalNodes[i];
            int numMutations = this.mutations.getMutations(node.getLeft()).size() + this.mutations.getMutations(node.getRight()).size();

            int band = 0;
            while (band < this.bounds.length && numMutations > this.bounds[band]) {
                band++;
            }
            bandOfNode[i] = band;
        }
        return bandOfNode;
    }

    @Override
    protected String getBandDescription() {
        return "mutation bands (scale factor, acceptance):";
    }

}

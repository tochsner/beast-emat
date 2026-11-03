package emat.initalisation;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.inference.StateNodeInitialiser;
import beast.base.spec.inference.parameter.RealScalarParam;

import java.util.Arrays;
import java.util.List;

@Description("Initialises the clock rate from a root-to-tip regression on a starting tree. The tips are placed " +
        "oldest first next to their closest placed tip, the branch lengths are the Fitch parsimony changes per " +
        "site, and the root is placed where the regression of the root-to-tip distances on the sampling dates " +
        "fits best. The slope is the clock rate in substitutions per site per time unit. The clock rate is left " +
        "unchanged if all tips have the same date or no root gives a positive slope.")
public class ClockRateInitialiser extends BEASTObject implements StateNodeInitialiser {

    final public Input<RealScalarParam<?>> clockRateInput = new Input<>("clockRate", "the clock rate to initialise", Input.Validate.REQUIRED);
    final public Input<Tree> treeInput = new Input<>("tree", "the tree whose tip dates give the sampling times", Input.Validate.REQUIRED);
    final public Input<Alignment> alignmentInput = new Input<>("alignment", "alignment of the tip sequences", Input.Validate.REQUIRED);

    @Override
    public void initAndValidate() {
        if (this.treeInput.get().getDateTrait() == null) {
            throw new IllegalArgumentException("The clock rate initialiser needs a date trait on the tree.");
        }
    }

    @Override
    public void initStateNodes() {
        double clockRate = this.computeClockRate();
        if (clockRate > 0.0) {
            this.clockRateInput.get().set(clockRate);
        }
    }

    /**
     * Returns the slope of the root-to-tip regression on the closest-placement tree with
     * parsimony branch lengths, or zero if all tips have the same date or no root gives a
     * positive slope.
     */
    double computeClockRate() {
        ClosestPlacement placement = new ClosestPlacement(this.treeInput.get(), this.alignmentInput.get());
        BinaryTopology topology = placement.computeTopology();
        PackedSequences sequences = placement.getSequences();

        int[] numChanges = sequences.countParsimonyChanges(topology);
        double[] branchLengths = Arrays.stream(numChanges).mapToDouble(changes -> (double) changes / sequences.getNumSites()).toArray();

        // the negated heights are the dates up to a constant, which does not change the slope
        double[] dates = Arrays.stream(placement.getTipHeights()).map(height -> -height).toArray();
        return new RootToTipRegression(topology, branchLengths, dates).computeBestRootSlope();
    }

    @Override
    public void getInitialisedStateNodes(List<StateNode> stateNodes) {
        stateNodes.add(this.clockRateInput.get());
    }

}

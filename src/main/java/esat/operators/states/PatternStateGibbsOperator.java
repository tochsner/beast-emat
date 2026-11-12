package esat.operators.states;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import emat.helper.FitchParsimony;
import esat.prior.GeneticPrior;
import esat.state.NodeStates;
import esat.stochasticmapping.GlobalStateSampler;
import esat.stochasticmapping.SubstitutionProcess;

import java.util.Arrays;

@Description("Resamples the states of every node at the sites of one pattern from their exact conditional " +
        "distribution given the tree, the model of the genetic prior, the tip data and the states at all other sites. " +
        "The pattern is drawn uniformly, or proportional to its Fitch parsimony score on the UPGMA tree of the " +
        "alignment plus a pseudo count, which only depends on the data. As a Gibbs move, it is always accepted. " +
        "A proposal costs a pruning pass over the tree for one pattern, independently of the number of sites with " +
        "that pattern.")
public class PatternStateGibbsOperator extends Operator {

    public enum PatternSelection {
        UNIFORM,
        PARSIMONY
    }

    final public Input<NodeStates> nodeStatesInput = new Input<>("nodeStates", "the node states to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<PatternSelection> patternSelectionInput = new Input<>("patternSelection", "the strategy for choosing the pattern: UNIFORM picks every pattern with the same probability, " +
            "PARSIMONY proportional to its Fitch parsimony score on the UPGMA tree of the alignment plus the pseudo count", PatternSelection.PARSIMONY, PatternSelection.values());
    final public Input<Double> pseudoCountInput = new Input<>("pseudoCount", "the weight added to the score of every pattern, so that patterns with a score of zero are also resampled. Not used by UNIFORM", 0.5);
    final public Input<Integer> maxNumSamplesInput = new Input<>("maxNumSamples", "the maximum number of sequences used to build the UPGMA tree for the parsimony scores. Larger alignments are uniformly subsampled", 1000);

    NodeStates nodeStates;
    SubstitutionProcess substitutionProcess;
    GlobalStateSampler sampler;

    // the cumulative selection weights of the patterns, or null for uniform selection
    double[] cumulativePatternWeights;

    @Override
    public void initAndValidate() {
        this.nodeStates = this.nodeStatesInput.get();
        this.substitutionProcess = new SubstitutionProcess(this.geneticPriorInput.get());
        this.sampler = new GlobalStateSampler(this.nodeStates, this.substitutionProcess);

        if (this.patternSelectionInput.get() == PatternSelection.PARSIMONY) {
            this.cumulativePatternWeights = this.computeCumulativePatternWeights();
        }
    }

    /**
     * Computes the cumulative selection weights of the patterns, each its Fitch parsimony score
     * on the UPGMA tree of the alignment plus the pseudo count. The scores are computed per
     * site, as the UPGMA tree may be built from a subsample with different patterns, and every
     * site of a pattern has the score of the pattern.
     */
    private double[] computeCumulativePatternWeights() {
        double pseudoCount = this.pseudoCountInput.get();
        if (pseudoCount < 0.0) {
            throw new IllegalArgumentException("The pseudo count must not be negative.");
        }

        Alignment alignment = this.nodeStates.getAlignment();
        int[] siteScores = FitchParsimony.computeOnUpgmaTree(alignment, this.maxNumSamplesInput.get()).getSiteScores();

        double[] patternWeights = new double[alignment.getPatternCount()];
        for (int site = 0; site < siteScores.length; site++) {
            patternWeights[alignment.getPatternIndex(site)] = siteScores[site] + pseudoCount;
        }

        double[] cumulativeWeights = new double[patternWeights.length];
        double totalWeight = 0.0;
        for (int patternNr = 0; patternNr < patternWeights.length; patternNr++) {
            totalWeight += patternWeights[patternNr];
            cumulativeWeights[patternNr] = totalWeight;
        }
        if (!(totalWeight > 0.0)) {
            throw new IllegalArgumentException("All patterns have zero weight. Use a positive pseudo count.");
        }
        return cumulativeWeights;
    }

    /**
     * Replaces the states at the sites of a drawn pattern by a draw from their joint
     * conditional distribution given everything else, so the Hastings ratio is infinite. The
     * pattern is drawn independently of the state, so the move is a Gibbs update.
     */
    @Override
    public double proposal() {
        this.substitutionProcess.update();
        int patternNr = this.drawPattern();
        GlobalStateSampler.PatternSample sample = this.sampler.samplePattern(patternNr);

        this.nodeStates.replaceStatesAtSites(sample.sites(), sample.rootStates(), sample.changesAtNode(), this);
        return Double.POSITIVE_INFINITY;
    }

    private int drawPattern() {
        if (this.cumulativePatternWeights == null) {
            return Randomizer.nextInt(this.sampler.getPatternCount());
        }

        double totalWeight = this.cumulativePatternWeights[this.cumulativePatternWeights.length - 1];
        int index = Arrays.binarySearch(this.cumulativePatternWeights, Randomizer.nextDouble() * totalWeight);

        // a negative index encodes the insertion point, the first pattern whose cumulative weight exceeds the draw
        return index >= 0 ? index + 1 : -index - 1;
    }

}

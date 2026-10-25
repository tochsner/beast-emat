package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import emat.operators.mutations.RootPathResampler;
import emat.prior.GeneticPrior;
import emat.stochasticmapping.HistorySampler;
import emat.stochasticmapping.SubstitutionProcess;

@Description("Scales the shorter of the two root branches by a factor drawn from a Bactrian kernel, which " +
        "moves the root height. By default, the mutations on the branches below the root move linearly along " +
        "with the root. Otherwise, the history of the path between the two root children is resampled for the " +
        "new root height by the chosen history sampler, so that the number of mutations on the root branches " +
        "follows their length, and the root sequence changes. Missations are not supported.")
public class RootScaleOperator extends ScaleTunedOperator {

    /** How the history on the branches below the root follows the root. */
    public enum PathHistory {
        LINEAR, JUKES_CANTOR, HYBRID
    }

    final public Input<Double> scaleUpperLimit = new Input<>("upper", "upper limit of the scale factor", 10.0);
    final public Input<Double> scaleLowerLimit = new Input<>("lower", "lower limit of the scale factor", 1e-8);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "width of the kernel in log space: close to zero is very small jumps, larger is larger jumps.", 0.1);
    final public Input<PathHistory> historySamplerInput = new Input<>("historySampler", "how the history on the branches below the root follows the root: LINEAR maps the mutations linearly along with the root, " +
            "JUKES_CANTOR and HYBRID resample the history of the path between the root children with the history sampler of that name",
            PathHistory.LINEAR, PathHistory.values());
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model, which resampling the history requires");

    // the model and the resampler of the path between the root children, or null if the mutations are mapped linearly
    SubstitutionProcess process;
    RootPathResampler rootPathResampler;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.scaleFactor = this.scaleFactorInput.get();
        this.upper = this.scaleUpperLimit.get();
        this.lower = this.scaleLowerLimit.get();

        PathHistory pathHistory = this.historySamplerInput.get();
        if (pathHistory != PathHistory.LINEAR) {
            GeneticPrior geneticPrior = this.geneticPriorInput.get();
            if (geneticPrior == null) {
                throw new IllegalArgumentException("Resampling the history requires the geneticPrior input.");
            }

            int numSites = this.mutations.getReferenceSequence().length;
            this.process = new SubstitutionProcess(geneticPrior, this.tree, this.mutations.getAlignment().getMaxStateCount());
            HistorySampler historySampler = pathHistory == PathHistory.HYBRID ? HistorySampler.HYBRID : HistorySampler.JUKES_CANTOR;
            this.rootPathResampler = new RootPathResampler(this.mutations, geneticPrior, this.process, historySampler.create(this.process, numSites));
        }
    }

    /**
     * Scales the shorter root branch h - h_m by a factor s, where h is the root height and h_m
     * the height of the higher child, which maps h to h_m + (h - h_m) s with Jacobian s. Every
     * mutation on the branch above a child c keeps its relative position on the branch, or the
     * history of the path between the root children is resampled for the new root height.
     */
    @Override
    public double proposal() {
        Node root = this.tree.getRoot();
        if (root.isLeaf()) {
            return Double.NEGATIVE_INFINITY;
        }

        double oldHeight = root.getHeight();
        double minHeight = this.computeMinHeight(root);
        double scaler = this.kernelDistribution.getScaler(root.getNr(), oldHeight, this.getCoercableParameterValue());

        // every positive scaler keeps the root above its children
        double newHeight = minHeight + (oldHeight - minHeight) * scaler;

        if (this.rootPathResampler != null) {
            this.process.update();
            return Math.log(scaler) + this.rootPathResampler.resample(root, newHeight, this);
        }

        root.setHeight(newHeight);

        double logHastingsRatio = Math.log(scaler);
        for (Node child : root.getChildren()) {
            logHastingsRatio += BranchTimeMapper.mapBranch(
                    this.mutations, child, oldHeight, child.getHeight(), newHeight, child.getHeight(), this
            );
        }

        return logHastingsRatio;
    }

    /** Returns the lowest height the root can take, which is above each child. */
    private double computeMinHeight(Node root) {
        double minHeight = Double.NEGATIVE_INFINITY;
        for (Node child : root.getChildren()) {
            minHeight = Math.max(minHeight, child.getHeight());
        }
        return minHeight;
    }

}

package emat.diagnostics;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Distribution;
import beast.base.inference.Operator;
import beast.base.spec.type.RealScalar;
import emat.state.Mutations;

import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPOutputStream;

@Description("Records one row per MCMC step for the study of how the chain moves along the ridge between the tree " +
        "(topology, node heights, mutation histories) and the tree prior parameters: the operator, whether the " +
        "proposal was accepted, the change of the log posterior and of its coalescent and genetic parts, the " +
        "proposed values of the parameters and tree statistics, a fingerprint of the proposed topology and the " +
        "measurements the operators annotate. The rows are written to a gzipped TSV file by the DiagnosticMCMC.")
public class ChainDiagnostics extends BEASTObject {

    final public Input<Tree> treeInput = new Input<>("tree", "the tree", Input.Validate.REQUIRED);
    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations on the tree", Input.Validate.REQUIRED);
    final public Input<Distribution> treePriorInput = new Input<>("treePrior", "the tree prior, e.g. the coalescent", Input.Validate.REQUIRED);
    final public Input<Distribution> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior of the mutations", Input.Validate.REQUIRED);
    final public Input<List<RealScalar<?>>> parametersInput = new Input<>("parameter", "the scalar parameters whose proposed values are recorded, e.g. the population size, the growth rate and the clock rate", new java.util.ArrayList<>());
    final public Input<String> fileNameInput = new Input<>("fileName", "the gzipped TSV file the rows are written to", Input.Validate.REQUIRED);

    final public Input<Integer> snapshotEveryInput = new Input<>("snapshotEvery", "if positive, the sorted internal node heights before and after every accepted step whose number is a multiple of this value are written to a gzipped binary file next to the step table", 0);

    final public Input<Integer> probeEveryInput = new Input<>("probeEvery", "if positive, the probes of the probed operators are drawn at every step whose number is a multiple of this value", 0);
    final public Input<Integer> probeCountInput = new Input<>("probeCount", "the number of proposals drawn per probed operator at a probe step", 100);

    final public Input<Boolean> profileInput = new Input<>("profile", "if true, the probe steps also write the profile of the log posterior along the line of a tree interval adapter", false);

    public static final int OUTCOME_ACCEPTED = 1;
    public static final int OUTCOME_REJECTED = 0;
    public static final int OUTCOME_REJECTED_IMPOSSIBLE = -1;
    public static final int OUTCOME_INVALID_PROPOSAL = -2;

    private static final int FLUSH_EVERY = 50_000;

    private Tree tree;
    private Mutations mutations;
    private Distribution treePrior;
    private Distribution geneticPrior;
    private List<RealScalar<?>> parameters;

    private PrintWriter out;
    private long numRows;
    private final StringBuilder row = new StringBuilder(512);

    // the random 64-bit hash of every leaf, from which the topology fingerprint is built
    private long[] leafHashes;

    // the parts of the log posterior and the topology in the last accepted state
    private double currentTreePriorLogP = Double.NaN;
    private double currentGeneticLogP = Double.NaN;
    private long currentFingerprint;

    private final double[] annotations = new double[StepAnnotations.NUM_SLOTS];

    // the sorted internal node heights before the step in progress, which is only filled on snapshot steps
    private DataOutputStream snapshots;
    private DataOutputStream probes;
    private DataOutputStream probeAnnotations;
    private DataOutputStream profiles;
    private float[] probeHeights;
    private float[] heightsBefore;
    private float[] heightsAfter;
    private boolean isSnapshotStep;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.mutations = this.mutationsInput.get();
        this.treePrior = this.treePriorInput.get();
        this.geneticPrior = this.geneticPriorInput.get();
        this.parameters = this.parametersInput.get();

        int numLeaves = this.tree.getLeafNodeCount();
        this.leafHashes = new long[numLeaves];
        for (int i = 0; i < numLeaves; i++) {
            this.leafHashes[i] = mix(0x9E3779B97F4A7C15L * (i + 1L));
        }

        try {
            this.out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(
                    new GZIPOutputStream(new FileOutputStream(this.fileNameInput.get()), 1 << 16, true), StandardCharsets.UTF_8), 1 << 16));
        } catch (IOException e) {
            throw new IllegalArgumentException("cannot open " + this.fileNameInput.get(), e);
        }
        if (this.probeEveryInput.get() > 0) {
            this.openProbes();
        }
        if (this.snapshotEveryInput.get() > 0) {
            this.openSnapshots();
        }
        Runtime.getRuntime().addShutdownHook(new Thread(this::close));
    }

    private void openSnapshots() {
        int numInternal = this.tree.getInternalNodeCount();
        this.heightsBefore = new float[numInternal];
        this.heightsAfter = new float[numInternal];
        try {
            this.snapshots = new DataOutputStream(new BufferedOutputStream(
                    new GZIPOutputStream(new FileOutputStream(this.fileNameInput.get() + ".heights.bin.gz"), 1 << 16, true), 1 << 16));
            this.snapshots.writeInt(numInternal);
        } catch (IOException e) {
            throw new IllegalArgumentException("cannot open the snapshot file of " + this.fileNameInput.get(), e);
        }
    }

    /**
     * Must be called before the proposal of every step. On the steps selected for snapshots it takes the
     * sorted internal node heights of the current state, which the accepted step is compared to.
     */
    public void prepareStep(long step) {
        this.isSnapshotStep = this.snapshots != null && step >= 0 && step % this.snapshotEveryInput.get() == 0;
        if (this.isSnapshotStep) {
            this.fillSortedInternalHeights(this.heightsBefore);
        }
    }

    private void openProbes() {
        int numInternal = this.tree.getInternalNodeCount();
        this.probeHeights = new float[numInternal];
        try {
            this.probes = new DataOutputStream(new BufferedOutputStream(
                    new GZIPOutputStream(new FileOutputStream(this.fileNameInput.get() + ".probes.bin.gz"), 1 << 16, true), 1 << 16));
            this.probes.writeInt(numInternal);
            this.probes.writeInt(this.probeCountInput.get());
            this.probeAnnotations = new DataOutputStream(new BufferedOutputStream(
                    new GZIPOutputStream(new FileOutputStream(this.fileNameInput.get() + ".probeannotations.bin.gz"), 1 << 16, true), 1 << 16));
        } catch (IOException e) {
            throw new IllegalArgumentException("cannot open the probe file of " + this.fileNameInput.get(), e);
        }
    }

    /** Returns whether the probe steps also write the profile of the posterior along a line. */
    public boolean isProfiled() {
        return this.profileInput.get();
    }

    /**
     * Starts a profile at the current state by writing the step and, for every node, whether it is a leaf, its parent
     * and its number of mutations.
     */
    public void beginProfile(long step) {
        try {
            if (this.profiles == null) {
                this.profiles = new DataOutputStream(new BufferedOutputStream(
                        new GZIPOutputStream(new FileOutputStream(this.fileNameInput.get() + ".profile.bin.gz"), 1 << 16, true), 1 << 16));
                this.profiles.writeInt(this.tree.getNodeCount());
            }
            this.profiles.writeLong(step);
            for (Node node : this.tree.getNodesAsArray()) {
                this.profiles.writeInt(node.isLeaf() ? 1 : 0);
                this.profiles.writeInt(node.isRoot() ? -1 : node.getParent().getNr());
                this.profiles.writeInt(this.mutations.getMutations(node).size());
            }
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * Writes one point of a profile: the offset along the line, the tree prior and the genetic prior of the state,
     * and for every node the contribution of its branch to the genetic prior and the length of the branch.
     */
    public void writeProfilePoint(double offset, double[] branchLogPs) {
        try {
            this.profiles.writeDouble(offset);
            this.profiles.writeDouble(this.treePrior.getCurrentLogP());
            this.profiles.writeDouble(this.geneticPrior.getCurrentLogP());
            for (Node node : this.tree.getNodesAsArray()) {
                this.profiles.writeFloat((float) branchLogPs[node.getNr()]);
            }
            for (Node node : this.tree.getNodesAsArray()) {
                this.profiles.writeFloat((float) (node.isRoot() ? 0.0 : node.getLength()));
            }
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Returns whether the probed operators draw their proposals at the given step. */
    public boolean isProbeStep(long step) {
        return this.probes != null && step >= 0 && step % this.probeEveryInput.get() == 0;
    }

    /** Starts the proposals of a probed operator at the current state by writing the step, the operator and the sorted internal node heights of the state. */
    public void beginProbe(long step, int operatorNr) {
        this.fillSortedInternalHeights(this.probeHeights);
        try {
            this.probes.writeLong(step);
            this.probes.writeInt(operatorNr);
            for (float height : this.probeHeights) {
                this.probes.writeFloat(height);
            }
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * Writes the log acceptance ratio of a probe proposal, the changes of the tree prior and the genetic prior (NaN if
     * the proposal was not evaluated) and the sorted internal node heights of the proposed state. The annotations of
     * the proposal go to a second file, in the same order and without headers, with one row of every slot per proposal.
     */
    public void writeProbe(double logAlpha, boolean isEvaluated) {
        this.fillSortedInternalHeights(this.probeHeights);
        StepAnnotations.drainInto(this.annotations);
        try {
            for (double annotation : this.annotations) {
                this.probeAnnotations.writeDouble(annotation);
            }
            this.probes.writeDouble(logAlpha);
            this.probes.writeDouble(isEvaluated ? this.treePrior.getCurrentLogP() - this.currentTreePriorLogP : Double.NaN);
            this.probes.writeDouble(isEvaluated ? this.geneticPrior.getCurrentLogP() - this.currentGeneticLogP : Double.NaN);
            for (float height : this.probeHeights) {
                this.probes.writeFloat(height);
            }
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    private void fillSortedInternalHeights(float[] target) {
        int i = 0;
        for (Node node : this.tree.getInternalNodes()) {
            target[i++] = (float) node.getHeight();
        }
        Arrays.sort(target);
    }

    /** Writes the header of the step table, with the annotation columns named after the given operators. */
    public void writeHeader() {
        StringBuilder header = new StringBuilder("step\toperator\toutcome\tlogAlpha\tlogHastings\tnanos\toldLogP\tnewLogP\tdTreePrior\tdGenetic\tdOtherPrior\t");
        for (int i = 0; i < this.parameters.size(); i++) {
            header.append("param").append(i).append('\t');
        }
        header.append("treeLength\trootHeight\tnumMutations\ttopology");
        for (int slot = 0; slot < StepAnnotations.NUM_SLOTS; slot++) {
            header.append("\ta").append(slot);
        }
        this.out.println(header);
    }

    /** Writes the operators of the chain with their annotation names next to the step table, and names the parameter columns. */
    public void writeLegend(List<Operator> operators) throws IOException {
        try (PrintWriter legend = new PrintWriter(this.fileNameInput.get() + ".legend.tsv", StandardCharsets.UTF_8)) {
            legend.println("kind\tindex\tid\tclass\tannotations");
            for (int i = 0; i < operators.size(); i++) {
                Operator operator = operators.get(i);
                String annotations = operator instanceof DiagnosedOperator diagnosed ? String.join(",", diagnosed.getAnnotationNames()) : "";
                legend.println("operator\t" + i + "\t" + operator.getID() + "\t" + operator.getClass().getSimpleName() + "\t" + annotations);
            }
            for (int i = 0; i < this.parameters.size(); i++) {
                legend.println("parameter\t" + i + "\t" + ((beast.base.core.BEASTInterface) this.parameters.get(i)).getID() + "\t\t");
            }
        }
    }

    /** Takes the tree prior, the genetic prior and the topology of the initial state as the last accepted state. */
    public void initialiseCurrentState() {
        this.currentTreePriorLogP = this.treePrior.getCurrentLogP();
        this.currentGeneticLogP = this.geneticPrior.getCurrentLogP();
        this.currentFingerprint = this.computeFingerprint();
    }

    /**
     * Records one step. This must be called after the proposal was evaluated and the operator was
     * told about the outcome, but before the state is restored, so that the state still holds the
     * proposal. The new log posterior is NaN if the proposal was never evaluated.
     */
    public void recordStep(long step, int operatorNr, Operator operator, int outcome, double logAlpha, double logHastings,
                           double oldLogP, double newLogP, long nanos) {
        StepAnnotations.drainInto(this.annotations);
        boolean isEvaluated = outcome != OUTCOME_INVALID_PROPOSAL;

        double dTreePrior = Double.NaN;
        double dGenetic = Double.NaN;
        double dOther = Double.NaN;
        long fingerprint = this.currentFingerprint;

        StringBuilder r = this.row;
        r.setLength(0);

        if (isEvaluated) {
            double treePriorLogP = this.treePrior.getCurrentLogP();
            double geneticLogP = this.geneticPrior.getCurrentLogP();
            dTreePrior = treePriorLogP - this.currentTreePriorLogP;
            dGenetic = geneticLogP - this.currentGeneticLogP;
            dOther = newLogP - oldLogP - dTreePrior - dGenetic;

            // only a subtree move can change the topology
            if (operator instanceof emat.operators.topology.SubtreePruneRegraftOperator) {
                fingerprint = this.computeFingerprint();
            }

            if (outcome == OUTCOME_ACCEPTED) {
                this.currentTreePriorLogP = treePriorLogP;
                this.currentGeneticLogP = geneticLogP;
                this.currentFingerprint = fingerprint;
            }
        }

        r.append(step).append('\t').append(operatorNr).append('\t').append(outcome).append('\t')
                .append(logAlpha).append('\t').append(logHastings).append('\t').append(nanos).append('\t')
                .append(oldLogP).append('\t').append(isEvaluated ? newLogP : Double.NaN).append('\t')
                .append(dTreePrior).append('\t').append(dGenetic).append('\t').append(dOther).append('\t');

        for (RealScalar<?> parameter : this.parameters) {
            r.append(isEvaluated ? parameter.get() : Double.NaN).append('\t');
        }

        r.append(isEvaluated ? this.computeTreeLength() : Double.NaN).append('\t')
                .append(isEvaluated ? this.tree.getRoot().getHeight() : Double.NaN).append('\t')
                .append(isEvaluated ? this.mutations.getMutationCount() : -1).append('\t')
                .append(Long.toHexString(fingerprint));
        for (double annotation : this.annotations) {
            r.append('\t');
            if (!Double.isNaN(annotation)) {
                r.append(annotation);
            }
        }
        this.out.println(r);

        if (this.isSnapshotStep && outcome == OUTCOME_ACCEPTED) {
            this.writeSnapshot(step, operatorNr);
        }

        this.numRows++;
        if (this.numRows % FLUSH_EVERY == 0) {
            this.out.flush();
        }
    }

    /** Writes the step, the operator and the sorted internal node heights before and after an accepted step. */
    private void writeSnapshot(long step, int operatorNr) {
        this.fillSortedInternalHeights(this.heightsAfter);
        try {
            this.snapshots.writeLong(step);
            this.snapshots.writeInt(operatorNr);
            for (float height : this.heightsBefore) {
                this.snapshots.writeFloat(height);
            }
            for (float height : this.heightsAfter) {
                this.snapshots.writeFloat(height);
            }
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Flushes and closes the files. */
    public synchronized void close() {
        if (this.out != null) {
            this.out.close();
            this.out = null;
        }
        if (this.profiles != null) {
            try {
                this.profiles.close();
            } catch (IOException ignored) {
            }
        }
        if (this.probeAnnotations != null) {
            try {
                this.probeAnnotations.close();
            } catch (IOException ignored) {
            }
        }
        if (this.probes != null) {
            try {
                this.probes.close();
            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
            this.probes = null;
        }
        if (this.snapshots != null) {
            try {
                this.snapshots.close();
            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
            this.snapshots = null;
        }
    }

    private double computeTreeLength() {
        double length = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                length += node.getParent().getHeight() - node.getHeight();
            }
        }
        return length;
    }

    /** Returns a 64-bit hash of the rooted topology, which does not depend on the order of the children. */
    private long computeFingerprint() {
        return this.hashSubtree(this.tree.getRoot());
    }

    private long hashSubtree(Node node) {
        if (node.isLeaf()) {
            return this.leafHashes[node.getNr()];
        }
        long hash = 0x2545F4914F6CDD1DL;
        for (Node child : node.getChildren()) {
            hash += this.hashSubtree(child);
        }
        return mix(hash);
    }

    private static long mix(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

}

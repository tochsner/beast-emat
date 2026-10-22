package emat.operators;

import adaptiveoperators.CenteredMultivariateNormalSampler;
import adaptiveoperators.ConditionalSampler;
import adaptiveoperators.MultivariateNormalSampler;
import adapters.Adapter;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Distribution;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import emat.helper.TreeSummaries;
import emat.state.Mutations;
import org.apache.commons.math4.legacy.linear.RealMatrix;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

@Description("Copy of the adaptiveoperators.AdaptiveOperator that also writes diagnostics. For every proposal it " +
        "writes the current mutable coordinates, the proposed change, the change of the tree length and root height, " +
        "the log Hastings ratio, the change of the log density of each monitored distribution and whether the " +
        "proposal was accepted. It also writes snapshots of the learned mean and covariance.")
public class DiagnosticAdaptiveOperator extends Operator {

    private static final double MIN_SCALE_FACTOR = 1.0E-3;
    private static final double MAX_SCALE_FACTOR = 1.0E3;

    // status codes of a proposal in the diagnostics file
    private static final int STATUS_ACCEPTED = 1;
    private static final int STATUS_REJECTED = 0;
    private static final int STATUS_REJECTED_ZERO_DENSITY = -1;
    private static final int STATUS_REJECTED_INVALID = -2;
    private static final int STATUS_UNKNOWN = -3;

    public final Input<List<Adapter>> adaptersInput = new Input<>("adapter", "", new ArrayList<>());
    public final Input<Tree> treeInput = new Input<>("tree", "");
    public final Input<Boolean> globalRedrawInput = new Input<>("globalRedraw",
            "If true, all mutable parameters of a node are redrawn from their learned conditional distribution " +
                    "given the immutable ones. If false, all mutable parameters are changed locally by adding " +
                    "zero-mean noise with the learned covariance to their current values.", true);
    public final Input<String> diagnosticsFileInput = new Input<>("diagnosticsFile",
            "File to which one line per proposal is written.", Input.Validate.REQUIRED);
    public final Input<String> snapshotFileInput = new Input<>("snapshotFile",
            "File to which the learned mean and covariance are written regularly.", Input.Validate.REQUIRED);
    public final Input<Integer> snapshotEveryInput = new Input<>("snapshotEvery",
            "Number of proposals between two snapshots.", 5_000);
    public final Input<Mutations> mutationsInput = new Input<>("mutations",
            "Mutations whose summaries are written for every proposal.");
    public final Input<List<Distribution>> monitoredInput = new Input<>("monitored",
            "Distributions whose log density change is written for every proposal.", new ArrayList<>());

    private List<Adapter> adapters;
    private Tree tree;
    private ConditionalSampler sampler;
    private boolean globalRedraw;
    private int numMutable;
    private int numImmutable;

    private final int burnIn = 1_000;
    private final int startTraining = 5_000;
    private final int endTraining = 400_000;
    private int count = 0;

    private double scaleFactor = 1.0;

    private List<Distribution> monitored;
    private BufferedWriter diagnosticsWriter;
    private BufferedWriter snapshotWriter;
    private int snapshotEvery;
    private ProposalRecord pending;
    private TreeSummaries summaries;

    @Override
    public void initAndValidate() {
        this.adapters = this.adaptersInput.get();
        this.tree = this.treeInput.get();
        this.monitored = this.monitoredInput.get();
        this.summaries = new TreeSummaries(this.tree, this.mutationsInput.get());
        this.snapshotEvery = this.snapshotEveryInput.get();

        int totalNumMutable = 0;
        int totalNumImmutable = 0;

        for (Adapter adapter : this.adapters) {
            totalNumMutable += adapter.getNumMutable();
            totalNumImmutable += adapter.getNumImmutable();
        }

        this.globalRedraw = this.globalRedrawInput.get();
        this.numMutable = totalNumMutable;
        this.numImmutable = totalNumImmutable;

        if (this.globalRedraw) {
            this.sampler = new MultivariateNormalSampler(totalNumImmutable, totalNumMutable);
        } else {
            this.sampler = new CenteredMultivariateNormalSampler(totalNumMutable);
        }

        this.openDiagnosticsFiles();
    }

    @Override
    public double proposal() {
        this.count++;

        // a proposal whose outcome was never reported belongs to a phase without bookkeeping
        this.flushPendingRecord(STATUS_UNKNOWN);

        if (this.count < this.burnIn) {
            // we are in burn in phase
            // we don't change the state nor record the state
            return 0;
        }

        int nodeId = Randomizer.nextInt(this.tree.getNodeCount());
        while (this.tree.getNode(nodeId).isLeaf() || this.tree.getNode(nodeId).isRoot()) {
            nodeId = Randomizer.nextInt(this.tree.getNodeCount());
        }

        double[] oldMutable = this.getMutable(nodeId);
        double[] oldImmutable = this.getImmutable(nodeId);

        // record the state in the sampler

        if (this.count < this.endTraining) {
            if (this.globalRedraw) {
                this.sampler.record(oldImmutable, oldMutable);
            } else {
                this.sampler.record(new double[0], oldMutable);
            }
        }

        if (this.count < this.startTraining) {
            // we are in the initial training phase
            // we don't change the state
            return 0;
        } else if (this.count == this.startTraining) {
            System.out.println("Start with adaptive kernel");
        } else if (this.count == this.endTraining) {
            System.out.println("End with adaptive kernel");
        }

        if (this.count % this.snapshotEvery == 0) {
            this.writeSnapshot();
        }

        ProposalRecord record = new ProposalRecord();
        record.call = this.count;
        record.scaleFactor = this.scaleFactor;
        record.oldMutable = oldMutable;
        record.oldLogTreeLength = Math.log(this.computeTreeLength());
        record.oldLogRootHeight = Math.log(this.tree.getRoot().getHeight());
        record.oldMonitored = this.getMonitoredLogDensities();
        record.oldSummaries = this.summaries.computeSummaries();

        // sample from the conditional distribution

        double[] proposal = this.sampleProposal(oldMutable, oldImmutable);

        record.delta = new double[this.numMutable];
        for (int i = 0; i < this.numMutable; i++) {
            record.delta[i] = proposal[i] - oldMutable[i];
        }

        // update the adapters

        double logDensityOld = 0.0;
        double logDensityNew = 0.0;

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            if (adapter.getNumMutable() == 0) continue;

            logDensityOld += adapter.getLogJacobianCorrection(nodeId);

            double[] proposedMutable = new double[adapter.getNumMutable()];
            System.arraycopy(proposal, idx, proposedMutable, 0, adapter.getNumMutable());
            adapter.update(proposedMutable, nodeId);

            logDensityNew += adapter.getLogJacobianCorrection(nodeId);

            idx += adapter.getNumMutable();
        }

        record.logJacobian = logDensityOld - logDensityNew;
        record.deltaLogTreeLength = Math.log(this.computeTreeLength()) - record.oldLogTreeLength;
        record.deltaLogRootHeight = Math.log(this.tree.getRoot().getHeight()) - record.oldLogRootHeight;

        double[] newSummaries = this.summaries.computeSummaries();
        record.deltaSummaries = new double[newSummaries.length];
        for (int i = 0; i < newSummaries.length; i++) {
            record.deltaSummaries[i] = newSummaries[i] - record.oldSummaries[i];
        }

        // compute and return the log hastings ratio

        double[] newImmutable = this.getImmutable(nodeId);
        if (!Arrays.stream(newImmutable).allMatch(Double::isFinite)) {
            // the proposal led to an invalid state, so we reject it
            this.pending = record;
            return Double.NEGATIVE_INFINITY;
        }

        if (this.globalRedraw) {
            // the local random walk is symmetric, so only the global redraw contributes proposal densities
            logDensityOld += this.sampler.logDensity(newImmutable, oldMutable, scaleFactor);
            logDensityNew += this.sampler.logDensity(oldImmutable, proposal, scaleFactor);
        }

        record.logHastings = logDensityOld - logDensityNew;
        this.pending = record;

        return record.logHastings;
    }

    private double[] sampleProposal(double[] oldMutable, double[] oldImmutable) {
        if (this.globalRedraw) {
            return this.sampler.sampleConditionally(oldImmutable, this.scaleFactor);
        }

        // change all mutable parameters locally
        double[] noise = this.sampler.sampleConditionally(new double[0], this.scaleFactor);
        double[] proposal = new double[this.numMutable];
        for (int i = 0; i < this.numMutable; i++) {
            proposal[i] = oldMutable[i] + noise[i];
        }

        return proposal;
    }

    private double[] getMutable(int nodeId) {
        double[] mutable = new double[this.numMutable];

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            double[] adapterMutable = adapter.getMutable(nodeId);
            System.arraycopy(adapterMutable, 0, mutable, idx, adapter.getNumMutable());
            idx += adapter.getNumMutable();
        }

        return mutable;
    }

    private double[] getImmutable(int nodeId) {
        double[] immutable = new double[this.numImmutable];

        int idx = 0;
        for (Adapter adapter : this.adapters) {
            double[] adapterImmutable = adapter.getImmutable(nodeId);
            System.arraycopy(adapterImmutable, 0, immutable, idx, adapter.getNumImmutable());
            idx += adapter.getNumImmutable();
        }

        return immutable;
    }

    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> nodes = new ArrayList<>();

        for (Adapter adapter : this.adapters) {
            nodes.addAll(adapter.listStateNodes());
        }

        return nodes;
    }

    @Override
    public double getCoercableParameterValue() {
        return this.scaleFactor;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        if (!Double.isFinite(value) || value <= 0.0) {
            throw new IllegalArgumentException("scaleFactor must be finite and positive");
        }
        this.scaleFactor = value;
    }

    @Override
    public void optimize(double logAlpha) {
        // the global redraw uses the learned conditional distribution as is
        if (this.globalRedraw) return;
        if (this.count < this.startTraining) return;

        double delta = this.calcDelta(logAlpha);
        delta += Math.log(this.scaleFactor);

        if (Double.isFinite(Math.exp(delta))) {
            this.scaleFactor = Math.min(MAX_SCALE_FACTOR, Math.max(MIN_SCALE_FACTOR, Math.exp(delta)));
        }
    }

    // outcome of a proposal

    @Override
    public void accept() {
        super.accept();
        this.flushPendingRecord(STATUS_ACCEPTED);
    }

    @Override
    public void reject(int reason) {
        super.reject(reason);

        if (this.pending == null) return;

        // the distributions still hold the log densities of the rejected state until the state is restored
        int status = switch (reason) {
            case -1 -> STATUS_REJECTED_ZERO_DENSITY;
            case -2 -> STATUS_REJECTED_INVALID;
            default -> STATUS_REJECTED;
        };
        this.flushPendingRecord(status);
    }

    // diagnostics

    /**
     * The state of one proposal, completed once the outcome is known.
     */
    private static class ProposalRecord {
        int call;
        double scaleFactor;
        double[] oldMutable;
        double[] delta;
        double oldLogTreeLength;
        double deltaLogTreeLength;
        double oldLogRootHeight;
        double deltaLogRootHeight;
        double logJacobian;
        double logHastings = Double.NEGATIVE_INFINITY;
        double[] oldMonitored;
        double[] oldSummaries;
        double[] deltaSummaries;
    }

    private void flushPendingRecord(int status) {
        ProposalRecord record = this.pending;
        if (record == null) return;
        this.pending = null;

        double[] newMonitored = status == STATUS_ACCEPTED || status == STATUS_REJECTED || status == STATUS_REJECTED_ZERO_DENSITY
                ? this.getMonitoredLogDensities()
                : record.oldMonitored;

        StringBuilder line = new StringBuilder();
        this.appendValue(line, record.call);
        this.appendValue(line, status);
        this.appendValue(line, record.scaleFactor);
        for (double value : record.oldMutable) this.appendValue(line, value);
        for (double value : record.delta) this.appendValue(line, value);
        this.appendValue(line, record.oldLogTreeLength);
        this.appendValue(line, record.deltaLogTreeLength);
        this.appendValue(line, record.oldLogRootHeight);
        this.appendValue(line, record.deltaLogRootHeight);
        this.appendValue(line, record.logJacobian);
        this.appendValue(line, record.logHastings);
        for (int i = 0; i < newMonitored.length; i++) {
            this.appendValue(line, newMonitored[i] - record.oldMonitored[i]);
        }
        for (double value : record.oldSummaries) this.appendValue(line, value);
        for (double value : record.deltaSummaries) this.appendValue(line, value);
        line.setLength(line.length() - 1);

        this.writeLine(this.diagnosticsWriter, line.toString());
    }

    private void appendValue(StringBuilder line, double value) {
        line.append(String.format(Locale.ROOT, "%.6g", value)).append('\t');
    }

    private double[] getMonitoredLogDensities() {
        double[] logDensities = new double[this.monitored.size()];
        for (int i = 0; i < logDensities.length; i++) {
            logDensities[i] = this.monitored.get(i).getCurrentLogP();
        }
        return logDensities;
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

    /**
     * Writes the call number, the scale factor, the number of recorded states, the mean and the full learned
     * covariance on one line.
     */
    private void writeSnapshot() {
        if (!(this.sampler instanceof CenteredMultivariateNormalSampler centered)) return;

        RealMatrix covariance = centered.getCovariance();
        StringBuilder line = new StringBuilder();
        this.appendValue(line, this.count);
        this.appendValue(line, this.scaleFactor);
        for (double value : centered.getMean()) this.appendValue(line, value);
        for (int i = 0; i < this.numMutable; i++) {
            for (int j = 0; j < this.numMutable; j++) {
                this.appendValue(line, covariance.getEntry(i, j));
            }
        }
        line.setLength(line.length() - 1);

        this.writeLine(this.snapshotWriter, line.toString());
    }

    private void openDiagnosticsFiles() {
        try {
            this.diagnosticsWriter = Files.newBufferedWriter(Path.of(this.diagnosticsFileInput.get()));
            this.snapshotWriter = Files.newBufferedWriter(Path.of(this.snapshotFileInput.get()));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

        this.writeLine(this.diagnosticsWriter, this.buildDiagnosticsHeader());

        Runtime.getRuntime().addShutdownHook(new Thread(this::closeDiagnosticsFiles));
    }

    private String buildDiagnosticsHeader() {
        List<String> names = new ArrayList<>(List.of("call", "status", "scale"));
        for (int i = 0; i < this.numMutable; i++) names.add("old" + i);
        for (int i = 0; i < this.numMutable; i++) names.add("delta" + i);
        names.addAll(List.of("oldLogTreeLength", "deltaLogTreeLength", "oldLogRootHeight", "deltaLogRootHeight", "logJacobian", "logHastings"));
        for (int i = 0; i < this.monitored.size(); i++) names.add("deltaMonitored" + i);
        for (String name : TreeSummaries.NAMES) names.add("old_" + name);
        for (String name : TreeSummaries.NAMES) names.add("delta_" + name);
        return String.join("\t", names);
    }

    private void writeLine(BufferedWriter writer, String line) {
        try {
            writer.write(line);
            writer.newLine();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private synchronized void closeDiagnosticsFiles() {
        try {
            this.diagnosticsWriter.close();
            this.snapshotWriter.close();
        } catch (IOException e) {
            // nothing sensible to do while the process shuts down
        }
    }

}

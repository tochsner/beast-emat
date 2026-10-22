package emat.benchmark;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.Operator;
import beast.base.spec.type.RealScalar;
import emat.helper.TreeSummaries;
import emat.state.Mutations;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Description("An MCMC chain that writes, for every step that changed the state, the operator that made the step and " +
        "the values of the clock rate, the tree prior parameters and a set of tree summaries after the step. The " +
        "changes of every summary can then be attributed to the operators that made them.")
public class DiagnosticMCMC extends TimedMCMC {

    private static final String[] TRACKED_SUMMARIES = {
            "logTreeLength", "logCoalescentExposure", "logLineagesAt10", "logLineagesAt25", "logSecondRootInterval",
            "logRootHeight", "logMutationCount", "logMeanMutationHeight", "logInternalLength", "logExternalLength"
    };

    public final Input<Mutations> mutationsInput = new Input<>("mutations", "The mutations to summarise.", Input.Validate.REQUIRED);
    public final Input<RealScalar> clockRateInput = new Input<>("clockRate", "The clock rate.", Input.Validate.REQUIRED, RealScalar.class);
    public final Input<RealScalar> populationSizeInput = new Input<>("populationSize", "The population size.", Input.Validate.REQUIRED, RealScalar.class);
    public final Input<RealScalar> growthRateInput = new Input<>("growthRate", "The growth rate.", Input.Validate.REQUIRED, RealScalar.class);
    public final Input<String> diagnosticsPrefixInput = new Input<>("diagnosticsPrefix",
            "Prefix of the binary file of steps and the text file of operator names.", Input.Validate.REQUIRED);

    private TreeSummaries summaries;
    private int[] trackedIndices;
    private DataOutputStream stepWriter;

    private final Map<String, Integer> operatorIndices = new HashMap<>();
    private final List<String> operatorNames = new ArrayList<>();

    private double[] previousValues;
    private double previousLogLikelihood = Double.NaN;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.summaries = new TreeSummaries(this.treeInput.get(), this.mutationsInput.get());

        List<String> names = List.of(TreeSummaries.NAMES);
        this.trackedIndices = new int[TRACKED_SUMMARIES.length];
        for (int i = 0; i < TRACKED_SUMMARIES.length; i++) {
            this.trackedIndices[i] = names.indexOf(TRACKED_SUMMARIES[i]);
        }

        try {
            this.stepWriter = new DataOutputStream(new BufferedOutputStream(
                    Files.newOutputStream(Path.of(this.diagnosticsPrefixInput.get() + ".steps.bin")), 1 << 20));
        } catch (IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    @Override
    public void run() throws IOException, SAXException, ParserConfigurationException {
        super.run();
        this.closeDiagnostics();
    }

    /**
     * Performs a step like the base class and, if it changed the state, writes the step number, the operator, the
     * posterior and the new values of all tracked quantities. The values before the first step are written with
     * the operator index -1.
     */
    @Override
    protected Operator propagateState(final long sampleNr) {
        if (this.previousValues == null) {
            this.previousValues = this.readValues();
            this.writeStep(-1, -1, this.previousValues, this.oldLogLikelihood);
        }

        Operator operator = super.propagateState(sampleNr);

        // a rejected step restores the state, so only steps that changed a tracked value are written
        double[] values = this.readValues();
        boolean changed = this.oldLogLikelihood != this.previousLogLikelihood && !Double.isNaN(this.previousLogLikelihood);
        for (int i = 0; i < values.length && !changed; i++) {
            changed = values[i] != this.previousValues[i];
        }
        this.previousLogLikelihood = this.oldLogLikelihood;

        if (changed && sampleNr >= 0) {
            this.writeStep(sampleNr, this.getOperatorIndex(operator), values, this.oldLogLikelihood);
        }
        this.previousValues = values;

        return operator;
    }

    private double[] readValues() {
        double[] all = this.summaries.computeSummaries();
        double[] values = new double[this.trackedIndices.length + 3];
        for (int i = 0; i < this.trackedIndices.length; i++) {
            values[i] = all[this.trackedIndices[i]];
        }
        values[this.trackedIndices.length] = Math.log(this.clockRateInput.get().get());
        values[this.trackedIndices.length + 1] = Math.log(this.populationSizeInput.get().get());
        values[this.trackedIndices.length + 2] = this.growthRateInput.get().get();
        return values;
    }

    private int getOperatorIndex(Operator operator) {
        return this.operatorIndices.computeIfAbsent(operator.getName(), name -> {
            this.operatorNames.add(name);
            return this.operatorNames.size() - 1;
        });
    }

    private void writeStep(long sampleNr, int operatorIndex, double[] values, double logLikelihood) {
        try {
            this.stepWriter.writeLong(sampleNr);
            this.stepWriter.writeShort(operatorIndex);
            this.stepWriter.writeDouble(logLikelihood);
            for (double value : values) {
                this.stepWriter.writeDouble(value);
            }
        } catch (IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    private void closeDiagnostics() throws IOException {
        this.stepWriter.close();

        List<String> lines = new ArrayList<>(this.operatorNames);
        lines.add("#values: " + String.join(",", TRACKED_SUMMARIES) + ",logClockRate,logPopulationSize,growthRate");
        Files.write(Path.of(this.diagnosticsPrefixInput.get() + ".ops.txt"), lines);
    }

}

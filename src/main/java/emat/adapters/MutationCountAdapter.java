package emat.adapters;

import adapters.Adapter;
import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.StateNode;
import emat.state.Mutations;

import java.util.List;

@Description("Immutable adapter that summarises the mutation history by the number of mutations on the tree.")
public class MutationCountAdapter extends BEASTObject implements Adapter {

    public final Input<Mutations> mutationsInput = new Input<>("mutations", "Mutations to count.", Input.Validate.REQUIRED);

    private Mutations mutations;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
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
        return new double[]{this.mutations.getMutationCount()};
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
        return List.of(this.mutations);
    }
}

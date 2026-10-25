package emat.helper;

import beast.base.evolution.tree.Node;
import beast.base.inference.Operator;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/** Changes the root sequence of an EMAT, which the mutations above the root encode as its difference to the reference sequence. */
public final class RootSequence {

    private RootSequence() {
    }

    /**
     * Sets the root state of the given sites. The mutations above the root are rebuilt with a
     * single mutation per site where the root differs from the reference sequence. They are
     * left untouched if no root state changes.
     */
    public static void setRootStates(Mutations mutations, Node root, Map<Integer, Integer> newRootStates, Operator operator) {
        int[] referenceSequence = mutations.getReferenceSequence();

        Map<Integer, Integer> rootStates = new TreeMap<>();
        for (Mutation mutation : mutations.getMutations(root)) {
            rootStates.put(mutation.site(), mutation.newState());
        }

        boolean isChanged = false;
        for (Map.Entry<Integer, Integer> entry : newRootStates.entrySet()) {
            int site = entry.getKey();
            int oldState = rootStates.getOrDefault(site, referenceSequence[site]);
            if (oldState != entry.getValue()) {
                rootStates.put(site, entry.getValue());
                isChanged = true;
            }
        }

        if (!isChanged) {
            return;
        }

        // the times of the mutations above the root are meaningless, so they are set to the root height

        double rootHeight = root.getHeight();
        List<Mutation> rootMutations = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : rootStates.entrySet()) {
            int site = entry.getKey();
            if (entry.getValue() != referenceSequence[site]) {
                rootMutations.add(new Mutation(root.getNr(), rootHeight, rootHeight, site, referenceSequence[site], entry.getValue()));
            }
        }

        mutations.applyMutations(root, rootMutations, operator);
    }

}

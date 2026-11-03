package emat.diagnostics;

/** An operator that writes measurements of its proposals to {@link StepAnnotations}. */
public interface DiagnosedOperator {

    /** Returns the names of the annotation slots this operator writes, in slot order. */
    String[] getAnnotationNames();

}

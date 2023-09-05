// added to allow compilation
// Snippet s15
// SNIPPET_STARTS
public static void s15() {
    int stepSize = Math.min((option.getMaximumValue() - option.getMinimumValue()) / 10, 1000);
    spinner = new JSpinner(new SpinnerNumberModel(option.getValue(), option.getMinimumValue(), option.getMaximumValue(), Math.max(1, stepSize)));
    // rename getShortDescription to toString to allow compilation
    spinner.setToolTipText(option.toString());
}
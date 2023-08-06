package snippet_splitter_out.ds_3;
public class ds_3_snip_83_s15 {
public static void s15() {
        int stepSize = Math.min((option.getMaximumValue() - option.getMinimumValue()) / 10, 1000);
        spinner = new JSpinner(new SpinnerNumberModel(option.getValue(), option.getMinimumValue(),
                option.getMaximumValue(), Math.max(1, stepSize)));
        spinner.setToolTipText(option.toString()); // rename getShortDescription to toString to allow compilation
    }
}
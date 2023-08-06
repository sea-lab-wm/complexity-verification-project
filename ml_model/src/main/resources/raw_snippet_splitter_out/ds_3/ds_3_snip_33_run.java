package snippet_splitter_out.ds_3;
public class ds_3_snip_33_run {
public void run(RunNotifier notifier) {
        TestResult result= new TestResult();
        result.addListener(createAdaptingListener(notifier));
        fTest.run(result);
    }
}
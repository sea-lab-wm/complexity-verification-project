import java.io.File;

public class Parser_New {
    private static final String FILE_PATH = "simple-datasets/src/main/java/cog_complexity_validation_datasets/One";
    public static void main(String[] args) {
        File projectDir = new File(FILE_PATH);
        FeatureExtractor.listIfStatements(projectDir);
        FeatureExtractor.listParameters(projectDir);
        FeatureExtractor.listLoopStatements(projectDir);
    }
}

package FeatureExtraction;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileNotFoundException;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class FeatureExtractorTest {

    private FeatureExtractor.FeatureVisitor featureVisitor;

    final int NUM_OF_LOOP_STATEMENTS = 9;
    final int NUM_OF_IF_STATEMENTS = 6;
    final int NUM_OF_PARAMETERS = 2;

    @BeforeEach
    public void setup(){
        File file = new File(getClass().getResource("/data/TestSnippet_1.java").getFile());
        CompilationUnit cu = null;
        try {
            cu = StaticJavaParser.parse(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        featureVisitor = new FeatureExtractor().new FeatureVisitor();
        featureVisitor.visit(cu,null);
    }

    @Test
    public void testLoops() {
        assertEquals(NUM_OF_LOOP_STATEMENTS, featureVisitor.getFeatures().getNumOfLoops());
    }

    @Test
    public void testIfStatements() {
        assertEquals(NUM_OF_IF_STATEMENTS, featureVisitor.getFeatures().getNumOfIfStatements());
    }

    @Test
    public void testMethodParameters() {
        assertEquals(NUM_OF_PARAMETERS, featureVisitor.getFeatures().getNumOfParameters());
    }

}

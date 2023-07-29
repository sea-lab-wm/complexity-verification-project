package edu.wm.sealab.featureextraction;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import java.io.File;
import java.io.FileNotFoundException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class FeatureExtractorTestMultiple {

  private FeatureVisitor featureVisitor1;
  private FeatureVisitor featureVisitor2;

  final int NUM_OF_LOOP_STATEMENTS_1 = 9;
  final int NUM_OF_IF_STATEMENTS_1 = 6;
  final int NUM_OF_PARAMETERS_1 = 2;
  final int NUM_OF_COMMENTS_1 = 7;

  final int NUM_OF_LOOP_STATEMENTS_2 = 6;
  final int NUM_OF_IF_STATEMENTS_2 = 4;
  final int NUM_OF_PARAMETERS_2 = 3;
  final int NUM_OF_COMMENTS_2 = 3;

  @BeforeEach
  public void setup() {
    System.out.println("Working Directory = " + System.getProperty("user.dir"));

    String dirPath = "src/test/resources/data";
    File projectDir = new File(dirPath);

    new DirExplorer(
            (level, path, file) -> path.endsWith(".java"),
            (level, path, file) -> {
              // File file = new File(getClass().getResource("/data/TestSnippet_1.java").getFile());
              // File file = new File("src/test/resources/data/TestSnippet_1.java");
              System.out.println(file.getName());

              CompilationUnit cu = null;
              try {
                cu = StaticJavaParser.parse(file);
              } catch (FileNotFoundException e) {
                e.printStackTrace();
              }

              if (file.getName().equals("TestSnippet_1.java")) {
                featureVisitor1 = new FeatureVisitor();
                featureVisitor1.visit(cu, null);
              } else if (file.getName().equals("TestSnippet_2.java")) {
                featureVisitor2 = new FeatureVisitor();
                featureVisitor2.visit(cu, null);
              }
            })
        .explore(projectDir);
  }

  @Test
  public void testLoops1() {
    assertEquals(NUM_OF_LOOP_STATEMENTS_1, featureVisitor1.getFeatures().getNumOfLoops());
  }

  @Test
  public void testIfStatements1() {
    assertEquals(NUM_OF_IF_STATEMENTS_1, featureVisitor1.getFeatures().getNumOfIfStatements());
  }

  @Test
  public void testMethodParameters1() {
    assertEquals(NUM_OF_PARAMETERS_1, featureVisitor1.getFeatures().getNumOfParameters());
  }

  @Test
  public void testComments1() {
    assertEquals(NUM_OF_COMMENTS_1, featureVisitor1.getFeatures().getNumOfComments());
  }

  @Test
  public void testLoops2() {
    assertEquals(NUM_OF_LOOP_STATEMENTS_2, featureVisitor2.getFeatures().getNumOfLoops());
  }

  @Test
  public void testIfStatements2() {
    assertEquals(NUM_OF_IF_STATEMENTS_2, featureVisitor2.getFeatures().getNumOfIfStatements());
  }

  @Test
  public void testMethodParameters2() {
    assertEquals(NUM_OF_PARAMETERS_2, featureVisitor2.getFeatures().getNumOfParameters());
  }

  @Test
  public void testComments2() {
    assertEquals(NUM_OF_COMMENTS_2, featureVisitor2.getFeatures().getNumOfComments());
  }
}

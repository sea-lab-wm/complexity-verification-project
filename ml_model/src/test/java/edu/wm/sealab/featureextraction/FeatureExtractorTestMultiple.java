package edu.wm.sealab.featureextraction;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParserConfiguration;
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
  final int NUM_OF_COMMENTS_1 = 9;
  final int NUM_OF_LINES_OF_CODE_1 = 50;
  final int NUM_OF_COMPARISONS_1 = 9;
  final int NUM_OF_ARITHMETIC_OPERATORS_1 = 3;
  final int NUM_OF_CONDITIONALS_1 = 7;
  final int NUM_OF_ASSIGNMENT_EXPRESSIONS_1 = 4;
  final int NUM_OF_NUMBERS_1 = 24;
  final int MAX_NUMBERS_1 = 2;


  final int NUM_OF_LOOP_STATEMENTS_2 = 6;
  final int NUM_OF_IF_STATEMENTS_2 = 4;
  final int NUM_OF_PARAMETERS_2 = 3;
  final int NUM_OF_COMMENTS_2 = 7;
  final int NUM_OF_LINES_OF_CODE_2 = 36;
  final int NUM_OF_COMPARISONS_2 = 6;
  final int NUM_OF_ARITHMETIC_OPERATORS_2 = 5;
  final int NUM_OF_CONDITIONALS_2 = 5;
  final int NUM_OF_ASSIGNMENT_EXPRESSIONS_2 = 5;
  final int NUM_OF_NUMBERS_2 = 16;
  final int MAX_NUMBERS_2 = 2;



  static FeatureVisitor featureVisitor = null;

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

              featureVisitor = new FeatureVisitor();
              CompilationUnit cu = null;
              CompilationUnit cuNoComm = null;
              try {
                cu = StaticJavaParser.parse(file);
                JavaParser parser = new JavaParser(new ParserConfiguration().setAttributeComments(false));
                cuNoComm = parser.parse(file).getResult().get();
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

              featureVisitor.visit(cu, null);

              // Extract syntactic features (non JavaParser extraction)
              SyntacticFeatureExtractor sfe =
                new SyntacticFeatureExtractor(featureVisitor.getFeatures());
              FeatureExtractorTest.features = sfe.extract(cuNoComm.toString());
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
  public void testAvgComments1() {
    assertEquals(1.0 * NUM_OF_COMMENTS_1 / NUM_OF_LINES_OF_CODE_1, 1.0 * featureVisitor1.getFeatures().getNumOfComments() / NUM_OF_LINES_OF_CODE_1);
  }

  @Test
  public void testComparisons1() {
    assertEquals(NUM_OF_COMPARISONS_1, featureVisitor1.getFeatures().getComparisons());
  }

  @Test
  public void testArithmeticOperators1() {
    assertEquals(NUM_OF_ARITHMETIC_OPERATORS_1, featureVisitor1.getFeatures().getArithmeticOperators());
  }

  @Test
  public void testConditionals1() {
    assertEquals(NUM_OF_CONDITIONALS_1, featureVisitor1.getFeatures().getConditionals());
  }
  
  @Test
  public void testAvgLoops1() {
    assertEquals(1.0 * NUM_OF_LOOP_STATEMENTS_1 / NUM_OF_LINES_OF_CODE_1, 1.0 * featureVisitor1.getFeatures().getNumOfLoops() / NUM_OF_LINES_OF_CODE_1);
  }

  @Test
  public void testAvgAssignExprs1() {
    assertEquals(1.0 * NUM_OF_ASSIGNMENT_EXPRESSIONS_1 / NUM_OF_LINES_OF_CODE_1, 1.0 * featureVisitor1.getFeatures().getAssignExprs() / NUM_OF_LINES_OF_CODE_1);
  }

  @Test
  public void testAvgNumbers1() {
    assertEquals(1.0 * NUM_OF_NUMBERS_1 / NUM_OF_LINES_OF_CODE_1, 1.0 * featureVisitor1.getFeatures().getNumbers() / NUM_OF_LINES_OF_CODE_1);
  }

  @Test
  public void testMaxNumbers1() {
    assertEquals(MAX_NUMBERS_1, featureVisitor1.getFeatures().findMaxNumbers());
  }

    @Test
  public void testAvgConditionals1() {
    assertEquals(1.0 * NUM_OF_CONDITIONALS_1 / NUM_OF_LINES_OF_CODE_1, 1.0 * featureVisitor1.getFeatures().getConditionals() / NUM_OF_LINES_OF_CODE_1);
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

  @Test
  public void testAvgComments2() {
    assertEquals(1.0 * NUM_OF_COMMENTS_2 / NUM_OF_LINES_OF_CODE_2, 1.0 * featureVisitor2.getFeatures().getNumOfComments() / NUM_OF_LINES_OF_CODE_2);
  }

    @Test
  public void testComparisons2() {
    assertEquals(NUM_OF_COMPARISONS_2, featureVisitor2.getFeatures().getComparisons());
  }

  @Test
  public void testArithmeticOperators2() {
    assertEquals(NUM_OF_ARITHMETIC_OPERATORS_2, featureVisitor2.getFeatures().getArithmeticOperators());
  }

  @Test
  public void testConditionals2() {
    assertEquals(NUM_OF_CONDITIONALS_2, featureVisitor2.getFeatures().getConditionals());
  }
    
  @Test
  public void testAvgLoops2() {
    assertEquals(1.0 * NUM_OF_LOOP_STATEMENTS_2 / NUM_OF_LINES_OF_CODE_2, 1.0 * featureVisitor2.getFeatures().getNumOfLoops() / NUM_OF_LINES_OF_CODE_2);
  }

  @Test
  public void testAvgAssignExprs2() {
    assertEquals(1.0 * NUM_OF_ASSIGNMENT_EXPRESSIONS_2 / NUM_OF_LINES_OF_CODE_2, 1.0 * featureVisitor2.getFeatures().getAssignExprs() / NUM_OF_LINES_OF_CODE_2);
  }

  @Test
  public void testAvgNumbers2() {
    assertEquals(1.0 * NUM_OF_NUMBERS_2 / NUM_OF_LINES_OF_CODE_2, 1.0 * featureVisitor2.getFeatures().getNumbers() / NUM_OF_LINES_OF_CODE_2);
  }
  
  @Test
  public void testAverageConditionals2() {
    assertEquals(1.0 * NUM_OF_CONDITIONALS_2 / NUM_OF_LINES_OF_CODE_2, 1.0 * featureVisitor2.getFeatures().getConditionals() / NUM_OF_LINES_OF_CODE_2);
  }

  @Test
  public void testMaxNumbers2() {
    assertEquals(MAX_NUMBERS_2, featureVisitor2.getFeatures().findMaxNumbers());
  }
}

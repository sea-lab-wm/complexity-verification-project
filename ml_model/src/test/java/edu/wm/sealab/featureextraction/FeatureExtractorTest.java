package edu.wm.sealab.featureextraction;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class FeatureExtractorTest {

  private FeatureVisitor featureVisitor;
  SyntacticFeatureExtractor syntacticFeatureExtractor;

  final int NUM_OF_LOOP_STATEMENTS = 9;
  final int NUM_OF_IF_STATEMENTS = 6;
  final int NUM_OF_PARAMETERS = 2;
  final int NUM_OF_PARANTHESIS = 28;
  final int NUM_OF_COMMENTS = 9;
  final int NUM_OF_LINES_OF_CODE = 50;
  final int NUM_OF_COMPARISONS = 9;
  final int NUM_OF_ARITHMETIC_OPERATORS = 3;
  final int NUM_OF_CONDITIONALS = 7;
  final int NUM_OF_ASSIGNMENT_EXPRESSIONS = 4;
  final int NUM_OF_NUMBERS = 24;
  final int MAX_NUMBERS = 2;
  final int NUM_OF_STATEMENTS = 50;

  static Features features = null;

  @BeforeEach
  public void setup() {

    String filePath = "src/test/resources/data/TestSnippet_1.java";
    Path path = Paths.get(filePath);
    File file = new File(filePath);

    CompilationUnit cu = null;
    CompilationUnit cuNoComm = null;
    try {
      cu = StaticJavaParser.parse(file);
      JavaParser parser = new JavaParser(new ParserConfiguration().setAttributeComments(false));
      cuNoComm = parser.parse(file).getResult().get();
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }

    featureVisitor = new FeatureVisitor();

    // Capture Java Parser related features eg: #ifstmts
    featureVisitor.visit(cu, null);
    
    // Extract syntactic features (non JavaParser extraction)
    SyntacticFeatureExtractor sfe =
      new SyntacticFeatureExtractor(featureVisitor.getFeatures());
    FeatureExtractorTest.features = sfe.extract(cuNoComm.toString());  }

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

  @Test
  public void testParenthesis() {
    assertEquals(NUM_OF_PARANTHESIS, features.getParenthesis());
  }
    
  @Test
  public void testComments() {
    assertEquals(NUM_OF_COMMENTS, featureVisitor.getFeatures().getNumOfComments());
  }

  @Test
  public void testAvgComments() {
    assertEquals(1.0 * NUM_OF_COMMENTS / NUM_OF_LINES_OF_CODE, 1.0 * featureVisitor.getFeatures().getNumOfComments() / NUM_OF_LINES_OF_CODE);
  }

  @Test
  public void testComparisons() {
    assertEquals(NUM_OF_COMPARISONS, featureVisitor.getFeatures().getComparisons());
  }

  @Test
  public void testArithmeticOperators() {
    assertEquals(NUM_OF_ARITHMETIC_OPERATORS, featureVisitor.getFeatures().getArithmeticOperators());
  }

  @Test
  public void testConditionals() {
    assertEquals(NUM_OF_CONDITIONALS, featureVisitor.getFeatures().getConditionals());
  }

  @Test
  public void testAvgLoops() {
    assertEquals(1.0 * NUM_OF_LOOP_STATEMENTS / NUM_OF_LINES_OF_CODE, 1.0 * featureVisitor.getFeatures().getNumOfLoops() / NUM_OF_LINES_OF_CODE);
  }

  @Test
  public void testAvgAssignExprs() {
    assertEquals(1.0 * NUM_OF_ASSIGNMENT_EXPRESSIONS / NUM_OF_LINES_OF_CODE, 1.0 * featureVisitor.getFeatures().getAssignExprs() / NUM_OF_LINES_OF_CODE);
  }

  @Test
  public void testAvgNumbers() {
    assertEquals(1.0 * NUM_OF_NUMBERS / NUM_OF_LINES_OF_CODE, 1.0 * featureVisitor.getFeatures().getNumbers() / NUM_OF_LINES_OF_CODE);
  }

  @Test
  public void testMaxNumbers() {
    assertEquals(MAX_NUMBERS, featureVisitor.getFeatures().findMaxNumbers());
  }

  @Test
  public void testStatements() {
    assertEquals(NUM_OF_STATEMENTS, featureVisitor.getFeatures().getStatements());
  }
}

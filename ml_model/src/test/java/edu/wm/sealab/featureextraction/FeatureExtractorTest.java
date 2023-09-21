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

  final int NUM_OF_LOOP_STATEMENTS = 9;
  final int NUM_OF_IF_STATEMENTS = 6;
  final int NUM_OF_PARAMETERS = 2;
  final int NUM_OF_COMMENTS = 9;
  final int NUM_OF_LINES_OF_CODE = 50;
  final int NUM_OF_COMPARISONS = 9;
  final int NUM_OF_ARITHMETIC_OPERATORS = 3;
  final int NUM_OF_CONDITIONALS = 7;

  final int NUM_OF_PARANTHESIS = 56;
  final double AVG_NUM_OF_PARENTHESIS = 0.9491525292396545;
  final int NUM_OF_COMMAS = 2;
  final double AVG_NUM_OF_COMMAS = 0.033898305147886276;
  final int NUM_OF_PERIODS = 22;
  final double AVG_NUM_OF_PERIODS = 0.37288135290145874;
  final int NUM_OF_SPACES = 710;
  final double AVG_NUM_OF_SPACES = 12.03389835357666;
  //indentation length avg and max
  final int MAX_INDENTATION_LENGTH = 20;
  final double AVG_INDENTATION_LENGTH = 10.10169506072998;
  //line length avg and max
  final int MAX_LINE_LENGTH = 65;
  final double AVG_LINE_LENGTH = 24.86440658569336;
  //blank lines avg
  final double AVG_BLANK_LINES = 0.033898305147886276;

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
  public void testParenthesis() {
    assertEquals(NUM_OF_PARANTHESIS, features.getParenthesis());
  }

  @Test
  public void testAvgParenthesis() {
    assertEquals(AVG_NUM_OF_PARENTHESIS, features.getAvgParenthesis());
  }

  @Test
  public void testCommas() {
    assertEquals(NUM_OF_COMMAS, features.getCommas());
  }

  @Test
  public void testAvgCommas() {
    assertEquals(AVG_NUM_OF_COMMAS, features.getAvgCommas());
  }

  @Test
  public void testPeriods() {
    assertEquals(NUM_OF_PERIODS, features.getPeriods());
  }

  @Test
  public void testAvgPeriods() {
    assertEquals(AVG_NUM_OF_PERIODS, features.getAvgPeriods());
  }

  @Test
  public void testSpaces() {
    assertEquals(NUM_OF_SPACES, features.getSpaces());
  }

  @Test
  public void testAvgSpaces() {
    assertEquals(AVG_NUM_OF_SPACES, features.getAvgSpaces());
  }

  @Test
  public void testMaxIndentationLength() {
    assertEquals(MAX_INDENTATION_LENGTH, features.getMaxIndentation());
  }

  @Test
  public void testAvgIndentationLength() {
    assertEquals(AVG_INDENTATION_LENGTH, features.getAvgIndentation());
  }

  @Test
  public void testMaxLineLength() {
    assertEquals(MAX_LINE_LENGTH, features.getMaxLineLength());
  }

  @Test
  public void testAvgLineLength() {
    assertEquals(AVG_LINE_LENGTH, features.getAvgLineLength());
  }

  @Test
  public void testAvgBlankLength() {
    assertEquals(AVG_BLANK_LINES, features.getAvgBlankLines());
  }
}

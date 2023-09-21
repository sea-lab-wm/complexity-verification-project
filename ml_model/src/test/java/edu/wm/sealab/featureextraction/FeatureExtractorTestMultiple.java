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

  final int NUM_OF_PARANTHESIS_1 = 56;
  final double AVG_NUM_OF_PARENTHESIS_1 = 0.9491525292396545;
  final int NUM_OF_COMMAS_1 = 2;
  final double AVG_NUM_OF_COMMAS_1 = 0.033898305147886276;
  final int NUM_OF_PERIODS_1 = 22;
  final double AVG_NUM_OF_PERIODS_1 = 0.37288135290145874;
  final int NUM_OF_SPACES_1 = 710;
  final double AVG_NUM_OF_SPACES_1 = 12.03389835357666;
  //indentation length avg and max
  final int MAX_INDENTATION_LENGTH_1 = 20;
  final double AVG_INDENTATION_LENGTH_1 = 10.10169506072998;
  //line length avg and max
  final int MAX_LINE_LENGTH_1 = 65;
  final double AVG_LINE_LENGTH_1 = 24.86440658569336;
  //blank lines avg
  final double AVG_BLANK_LINES_1 = 0.033898305147886276;


  final int NUM_OF_LOOP_STATEMENTS_2 = 6;
  final int NUM_OF_IF_STATEMENTS_2 = 4;
  final int NUM_OF_PARAMETERS_2 = 3;
  final int NUM_OF_COMMENTS_2 = 7;
  final int NUM_OF_LINES_OF_CODE_2 = 36;
  final int NUM_OF_COMPARISONS_2 = 6;
  final int NUM_OF_ARITHMETIC_OPERATORS_2 = 5;
  final int NUM_OF_CONDITIONALS_2 = 5;

  final int NUM_OF_PARANTHESIS_2 = 40;
  final double AVG_NUM_OF_PARENTHESIS_2 = 0.8333333134651184;
  final int NUM_OF_COMMAS_2 = 3;
  final double AVG_NUM_OF_COMMAS_2 = 0.0625;
  final int NUM_OF_PERIODS_2 = 16;
  final double AVG_NUM_OF_PERIODS_2 = 0.3333333432674408;
  final int NUM_OF_SPACES_2 = 568;
  final double AVG_NUM_OF_SPACES_2 = 11.833333015441895;
  //indentation length avg and max
  final int MAX_INDENTATION_LENGTH_2 = 16;
  final double AVG_INDENTATION_LENGTH_2 = 9.833333015441895;
  //line length avg and max
  final int MAX_LINE_LENGTH_2 = 51;
  final double AVG_LINE_LENGTH_2 = 23.85416603088379;
  //blank lines avg
  final double AVG_BLANK_LINES_2 = 0.0416666679084301;



  static FeatureVisitor featureVisitor = null;

  static Features features1 = null; 
  static Features features2 = null; 

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
                SyntacticFeatureExtractor sfe1 = new SyntacticFeatureExtractor(featureVisitor1.getFeatures());
                features1 = sfe1.extract(cuNoComm.toString());
              } else if (file.getName().equals("TestSnippet_2.java")) {
                featureVisitor2 = new FeatureVisitor();
                featureVisitor2.visit(cu, null);
                SyntacticFeatureExtractor sfe2 = new SyntacticFeatureExtractor(featureVisitor2.getFeatures());
                features2 = sfe2.extract(cuNoComm.toString());
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
  public void testArithmeticOperators_1() {
    assertEquals(NUM_OF_ARITHMETIC_OPERATORS_1, featureVisitor1.getFeatures().getArithmeticOperators());
  }

  @Test
  public void testConditionals_1() {
    assertEquals(NUM_OF_CONDITIONALS_1, featureVisitor1.getFeatures().getConditionals());
  }

  @Test
  public void testParenthesis1() {
    assertEquals(NUM_OF_PARANTHESIS_1, features1.getParenthesis());
  }

  @Test
  public void testAvgParenthesis1() {
    assertEquals(AVG_NUM_OF_PARENTHESIS_1, features1.getAvgParenthesis());
  }

  @Test
  public void testCommas1() {
    assertEquals(NUM_OF_COMMAS_1, features1.getCommas());
  }

  @Test
  public void testAvgCommas1() {
    assertEquals(AVG_NUM_OF_COMMAS_1, features1.getAvgCommas());
  }

  @Test
  public void testPeriods1() {
    assertEquals(NUM_OF_PERIODS_1, features1.getPeriods());
  }

  @Test
  public void testAvgPeriods1() {
    assertEquals(AVG_NUM_OF_PERIODS_1, features1.getAvgPeriods());
  }

  @Test
  public void testSpaces1() {
    assertEquals(NUM_OF_SPACES_1, features1.getSpaces());
  }

  @Test
  public void testAvgSpaces1() {
    assertEquals(AVG_NUM_OF_SPACES_1, features1.getAvgSpaces());
  }

  @Test
  public void testMaxIndentationLength1() {
    assertEquals(MAX_INDENTATION_LENGTH_1, features1.getMaxIndentation());
  }

  @Test
  public void testAvgIndentationLength1() {
    assertEquals(AVG_INDENTATION_LENGTH_1, features1.getAvgIndentation());
  }

  @Test
  public void testMaxLineLength1() {
    assertEquals(MAX_LINE_LENGTH_1, features1.getMaxLineLength());
  }

  @Test
  public void testAvgLineLengt1h() {
    assertEquals(AVG_LINE_LENGTH_1, features1.getAvgLineLength());
  }

  @Test
  public void testAvgBlankLength1() {
    assertEquals(AVG_BLANK_LINES_1, features1.getAvgBlankLines());
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
  public void testArithmeticOperators_2() {
    assertEquals(NUM_OF_ARITHMETIC_OPERATORS_2, featureVisitor2.getFeatures().getArithmeticOperators());
  }

  @Test
  public void testConditionals_2() {
    assertEquals(NUM_OF_CONDITIONALS_2, featureVisitor2.getFeatures().getConditionals());
  }

  @Test
  public void testParenthesis2() {
    assertEquals(NUM_OF_PARANTHESIS_2, features2.getParenthesis());
  }

  @Test
  public void testAvgParenthesis2() {
    assertEquals(AVG_NUM_OF_PARENTHESIS_2, features2.getAvgParenthesis());
  }

  @Test
  public void testCommas2() {
    assertEquals(NUM_OF_COMMAS_2, features2.getCommas());
  }

  @Test
  public void testAvgCommas2() {
    assertEquals(AVG_NUM_OF_COMMAS_2, features2.getAvgCommas());
  }

  @Test
  public void testPeriods2() {
    assertEquals(NUM_OF_PERIODS_2, features2.getPeriods());
  }

  @Test
  public void testAvgPeriods2() {
    assertEquals(AVG_NUM_OF_PERIODS_2, features2.getAvgPeriods());
  }

  @Test
  public void testSpaces2() {
    assertEquals(NUM_OF_SPACES_2, features2.getSpaces());
  }

  @Test
  public void testAvgSpaces2() {
    assertEquals(AVG_NUM_OF_SPACES_2, features2.getAvgSpaces());
  }

  @Test
  public void testMaxIndentationLength2() {
    assertEquals(MAX_INDENTATION_LENGTH_2, features2.getMaxIndentation());
  }

  @Test
  public void testAvgIndentationLength2() {
    assertEquals(AVG_INDENTATION_LENGTH_2, features2.getAvgIndentation());
  }

  @Test
  public void testMaxLineLength2() {
    assertEquals(MAX_LINE_LENGTH_2, features2.getMaxLineLength());
  }

  @Test
  public void testAvgLineLength2() {
    assertEquals(AVG_LINE_LENGTH_2, features2.getAvgLineLength());
  }

  @Test
  public void testAvgBlankLength2() {
    assertEquals(AVG_BLANK_LINES_2, features2.getAvgBlankLines());
  }
}


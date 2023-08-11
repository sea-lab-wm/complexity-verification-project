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
  final int NUM_OF_PARANTHESIS = 27;
  final int NUM_OF_COMMENTS = 9;

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

    // Modify the CU to compute syntactic features i.e. parenthesis, commas, etc
    StringLiteralReplacer stringLiteralReplacer = new StringLiteralReplacer();
    stringLiteralReplacer.visit(cuNoComm, null);
    
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
}

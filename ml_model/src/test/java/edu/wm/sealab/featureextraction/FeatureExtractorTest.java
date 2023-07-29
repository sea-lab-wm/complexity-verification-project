package edu.wm.sealab.featureextraction;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.javaparser.StaticJavaParser;
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
  final int NUM_OF_COMMENTS = 7;

  @BeforeEach
  public void setup() {

    String filePath = "src/test/resources/data/TestSnippet_1.java";
    Path path = Paths.get(filePath);
    File file = new File(filePath);

    CompilationUnit cu = null;
    try {
      cu = StaticJavaParser.parse(file);
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }

    featureVisitor = new FeatureVisitor();

    syntacticFeatureExtractor = new SyntacticFeatureExtractor(featureVisitor.getFeatures());

    String codeSnippet = null;
    try {
      codeSnippet = Files.readAllLines(path).toString();
    } catch (IOException e) {
      e.printStackTrace();
    }

    // Capture Java Parser related features eg: #ifstmts
    featureVisitor.visit(cu, null);

    // Captures non-Java Parser related features eg: #parenthesis
    syntacticFeatureExtractor.extract(codeSnippet);
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

  @Test
  public void testParenthesis() {
    assertEquals(NUM_OF_PARANTHESIS, featureVisitor.getFeatures().getParenthesis());
  }

  @Test
  public void testComments() {
    assertEquals(NUM_OF_COMMENTS, featureVisitor.getFeatures().getNumOfComments());
  }
}

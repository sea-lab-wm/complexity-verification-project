package edu.wm.sealab.featureextraction;

import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.AssignExpr;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.BinaryExpr.Operator;
import com.github.javaparser.ast.expr.BooleanLiteralExpr;
import com.github.javaparser.ast.expr.CharLiteralExpr;
import com.github.javaparser.ast.expr.DoubleLiteralExpr;
import com.github.javaparser.ast.expr.IntegerLiteralExpr;
import com.github.javaparser.ast.expr.LongLiteralExpr;
import com.github.javaparser.ast.expr.NullLiteralExpr;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.expr.TextBlockLiteralExpr;
import com.github.javaparser.ast.stmt.AssertStmt;
import com.github.javaparser.ast.stmt.BreakStmt;
import com.github.javaparser.ast.stmt.CatchClause;
import com.github.javaparser.ast.stmt.ContinueStmt;
import com.github.javaparser.ast.stmt.DoStmt;
import com.github.javaparser.ast.stmt.ExplicitConstructorInvocationStmt;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.ForEachStmt;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.LabeledStmt;
import com.github.javaparser.ast.stmt.LocalClassDeclarationStmt;
import com.github.javaparser.ast.stmt.LocalRecordDeclarationStmt;
import com.github.javaparser.ast.stmt.ReturnStmt;
import com.github.javaparser.ast.stmt.SwitchStmt;
import com.github.javaparser.ast.stmt.SynchronizedStmt;
import com.github.javaparser.ast.stmt.ThrowStmt;
import com.github.javaparser.ast.stmt.TryStmt;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.github.javaparser.ast.stmt.YieldStmt;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import java.util.ArrayList;
import java.util.List;

/**
 * This class to extract features from a java file Input : Java file with a Single Method (Note: if
 * want to use all the features) Output: Features
 */
public class FeatureVisitor extends VoidVisitorAdapter<Void> {

  private Features features = new Features();

  /**
   * This method to compute #parameters of a java method
   */
  @Override
  public void visit(MethodDeclaration md, Void arg) {
    super.visit(md, arg);
    features.setNumOfParameters(md.getParameters().size());
  }

  /**
   * This method to compute #parameters of a java method
   */
  @Override
  public void visit(ConstructorDeclaration cd, Void arg) {
    super.visit(cd, arg);
    features.setNumOfParameters(cd.getParameters().size());
  }

  /**
   * This method to compute #if statements of a java method
   */
  @Override
  public void visit(IfStmt ifStmt, Void arg) {
    super.visit(ifStmt, arg);
    features.incrementNumOfIfStatements();
    features.incrementNumOfConditionals();
    features.incrementNumOfStatements();
  }

  /**
   * This method to compute # switch statements of a java method (not entries)
   */
  @Override
  public void visit(SwitchStmt swStmt, Void arg) {
    super.visit(swStmt, arg);
    features.incrementNumOfConditionals();
    features.incrementNumOfStatements();
  }

  /**
   * This method to compute # for loops of a java method
   */
  @Override
  public void visit(ForStmt forStmt, Void arg) {
    super.visit(forStmt, arg);
    features.incrementNumOfLoops();
    features.incrementNumOfStatements();
  }

  /**
   * This method to compute # while loops of a java method
   */
  @Override
  public void visit(WhileStmt whileStmt, Void arg) {
    super.visit(whileStmt, arg);
    features.incrementNumOfLoops();
    features.incrementNumOfStatements();
  }

  /**
   * This method to compute # for each loops of a java method
   */
  @Override
  public void visit(ForEachStmt forEachStmt, Void arg) {
    super.visit(forEachStmt, arg);
    features.incrementNumOfLoops();
    features.incrementNumOfStatements();
  }

  /**
   * This method computes # assignment expressions in a java method
   * Does not include declaration statements
   */
  @Override
  public void visit(AssignExpr assignExpr, Void arg) {
    super.visit(assignExpr, arg);
    features.setAssignExprs(features.getAssignExprs() + 1);
  }

  /**
   * This method computes # comparisons and arithmetic operators in a java method
   */
  @Override
  public void visit(BinaryExpr n, Void arg) {
    super.visit(n, arg);
    Operator operator = n.getOperator();
    if (operator == Operator.EQUALS
        || operator == Operator.NOT_EQUALS
        || operator == Operator.LESS
        || operator == Operator.LESS_EQUALS
        || operator == Operator.GREATER
        || operator == Operator.GREATER_EQUALS) {
      features.setComparisons(features.getComparisons() + 1);
    } else if (operator == Operator.PLUS
        || operator == Operator.MINUS
        || operator == Operator.MULTIPLY
        || operator == Operator.DIVIDE
        || operator == Operator.REMAINDER) {
      features.incrementNumOfArithmeticOperators();
    }
  }

  /**
   * This method identifies boolean literals in a java method and sums them up to the total number
   * of literals
   */
  @Override
  public void visit(BooleanLiteralExpr ble, Void arg) {
    super.visit(ble, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies char literals in a java method and sums them up to the total number of
   * literals
   */
  @Override
  public void visit(CharLiteralExpr cle, Void arg) {
    super.visit(cle, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies integer literals in a java method and sums them up to the total number
   * of literals
   */
  @Override
  public void visit(IntegerLiteralExpr ile, Void arg) {
    super.visit(ile, arg);
    features.incrementNumOfLiterals();
    features.incrementNumOfNumbers();
    String lineNumber = ile.getRange().get().begin.line+"";
    if(features.getLineNumberMap().containsKey(lineNumber)){
      List<Integer> list = features.getLineNumberMap().get(lineNumber);
      list.add(ile.asNumber().intValue());
      features.getLineNumberMap().put(lineNumber,list);
    }else{
      List<Integer> list = new ArrayList<>();
      list.add(ile.asNumber().intValue());
      features.getLineNumberMap().put(lineNumber,list);
    }
  }

  /**
   * This method identifies the long literals in a java method and sums them up to the total number
   * of literals
   */
  @Override
  public void visit(LongLiteralExpr lle, Void arg) {
    super.visit(lle, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies null literals in a java method and sums them up to the total number of
   * literals
   */
  @Override
  public void visit(NullLiteralExpr nle, Void arg) {
    super.visit(nle, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies string literals in a java method and sums them up to the total number of
   * literals
   */
  @Override
  public void visit(StringLiteralExpr sle, Void arg) {
    super.visit(sle, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies text block literals in a java method and sums them up to the total
   * number of literals
   */
  @Override
  public void visit(TextBlockLiteralExpr tble, Void arg) {
    super.visit(tble, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies double literals in a java method and sums them up to the total number of
   * literals
   */
  @Override
  public void visit(DoubleLiteralExpr dle, Void arg) {
    super.visit(dle, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies comments in a java method and sums them up to the total number of
   * comments. The ClassOrInterfaceDeclaration method getAllContainedComments() is used in order
   * to prevent orphan comments from being ignored by the Parser.
   */
  @Override
  public void visit(ClassOrInterfaceDeclaration cu, Void arg) {
    super.visit(cu, arg);
    features.setNumOfComments(cu.getAllContainedComments().size());
  }

  /**
   * This method identifies Assert Statements in a java method to sum up the total number of
   * all statements
   */
  @Override
  public void visit(AssertStmt ast, Void arg) {
    super.visit(ast, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method identifies Break Statements in a java method to sum up the total number of 
   * all statements
   */
  @Override
  public void visit(BreakStmt brst, Void arg) {
    super.visit(brst, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method identifies Catch Clauses in a java method to sum up the total number of 
   * all statements
   */
  @Override
  public void visit(CatchClause cc, Void arg) {
    super.visit(cc, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method identifies Continue Statements in a java method to sum up the total number of 
   * all statements
   */
  @Override
  public void visit(ContinueStmt cs, Void arg) {
    super.visit(cs, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method identifies Do Statements in a java method to sum up the total number of 
   * all statements
   */
  @Override
  public void visit(DoStmt ds, Void arg) {
    super.visit(ds, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method identifies Explicit Constructor Invocation Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(ExplicitConstructorInvocationStmt ecis, Void arg) {
    super.visit(ecis, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method identifies Expression Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(ExpressionStmt es, Void arg) {
    super.visit(es, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method identifies Labeled Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(LabeledStmt ls, Void arg) {
    super.visit(ls, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method identifies Local Class Declaration Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(LocalClassDeclarationStmt lcds, Void arg) {
    super.visit(lcds, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method identifies Local Record Declaration Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(LocalRecordDeclarationStmt lrds, Void arg) {
    super.visit(lrds, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method identifies Return Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(ReturnStmt rs, Void arg) {
    super.visit(rs, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method identifies Synchronized Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(SynchronizedStmt ss, Void arg) {
    super.visit(ss, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method identifies Throw Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(ThrowStmt ts, Void arg) {
    super.visit(ts, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method identifies Try Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(TryStmt ts, Void arg) {
    super.visit(ts, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method identifies Yield Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(YieldStmt ys, Void arg) {
    super.visit(ys, arg);
    features.incrementNumOfStatements();
  }

  /**
   * This method is to get the computed features After one/more visit method is/are called, the
   * features will be updated and then use this method to get the updated features
   */
  public Features getFeatures() {
    return features;
  }
}

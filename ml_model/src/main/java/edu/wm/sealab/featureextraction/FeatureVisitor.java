package edu.wm.sealab.featureextraction;

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
import com.github.javaparser.ast.stmt.ForEachStmt;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.comments.BlockComment;
import com.github.javaparser.ast.comments.JavadocComment;
import com.github.javaparser.ast.comments.LineComment;

/**
 * This class to extract features from a java file Input : Java file with a Single Method (Note: if
 * want to use all the features) Output: Features
 */
public class FeatureVisitor extends VoidVisitorAdapter<Void> {

  private Features features = new Features();

  /**
   * This method to compute #parameters of a java method
   *
   * @param MethodDeclaration
   * @param Void
   */
  @Override
  public void visit(MethodDeclaration md, Void arg) {
    super.visit(md, arg);
    features.setNumOfParameters(md.getParameters().size());
  }

  /**
   * This method to compute #parameters of a java method
   *
   * @param ConstructorDeclaration
   * @param Void
   */
  @Override
  public void visit(ConstructorDeclaration cd, Void arg) {
    super.visit(cd, arg);
    features.setNumOfParameters(cd.getParameters().size());
  }

  /**
   * This method to compute #if statements of a java method
   *
   * @param IfStmt
   * @param Void
   */
  @Override
  public void visit(IfStmt ifStmt, Void arg) {
    super.visit(ifStmt, arg);
    features.incrementNumOfIfStatements();
  }

  /**
   * This method to compute # for loops of a java method
   *
   * @param ForStmt
   * @param Void
   */
  @Override
  public void visit(ForStmt forStmt, Void arg) {
    super.visit(forStmt, arg);
    features.incrementNumOfLoops();
  }

  /**
   * This method to compute # while loops of a java method
   *
   * @param WhileStmt
   * @param Void
   */
  @Override
  public void visit(WhileStmt whileStmt, Void arg) {
    super.visit(whileStmt, arg);
    features.incrementNumOfLoops();
  }

  /**
   * This method to compute # for each loops of a java method
   *
   * @param ForEachStmt
   * @param Void
   */
  @Override
  public void visit(ForEachStmt forEachStmt, Void arg) {
    super.visit(forEachStmt, arg);
    features.incrementNumOfLoops();
  }

  /**
   * This method computes # assignment expressions in a java method
   *
   * @param AssignExpr
   * @param Void
   */
  @Override
  public void visit(AssignExpr assignExpr, Void arg) {
    super.visit(assignExpr, arg);
    features.setAssignExprs(features.getAssignExprs() + 1);
  }

  /**
   * This method computes # comparisons in a java method
   *
   * @param BinaryExpr
   * @param Void
   */
  @Override
  public void visit(BinaryExpr n, Void arg) {
    super.visit(n, arg);
    if (n.getOperator() == Operator.EQUALS
        || n.getOperator() == Operator.NOT_EQUALS
        || n.getOperator() == Operator.LESS
        || n.getOperator() == Operator.LESS_EQUALS
        || n.getOperator() == Operator.GREATER
        || n.getOperator() == Operator.GREATER_EQUALS) {
      features.setComparisons(features.getComparisons() + 1);
    }
  }

  /**
   * This method identifies boolean literals in a java method and sums them up to the total number
   * of literals
   *
   * @param BooleanLiteralExpr
   * @param Void
   */
  @Override
  public void visit(BooleanLiteralExpr ble, Void arg) {
    super.visit(ble, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies char literals in a java method and sums them up to the total number of
   * literals
   *
   * @param CHarLiteralExpr
   * @param Void
   */
  @Override
  public void visit(CharLiteralExpr cle, Void arg) {
    super.visit(cle, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies integer literals in a java method and sums them up to the total number
   * of literals
   *
   * @param IntegerLiteralExpr
   * @param Void
   */
  @Override
  public void visit(IntegerLiteralExpr ile, Void arg) {
    super.visit(ile, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies the long literals in a java method and sums them up to the total number
   * of literals
   *
   * @param LongLiteralExpr
   * @param Void
   */
  @Override
  public void visit(LongLiteralExpr lle, Void arg) {
    super.visit(lle, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies null literals in a java method and sums them up to the total number of
   * literals
   *
   * @param NullLiteralExpr
   * @param Void
   */
  @Override
  public void visit(NullLiteralExpr nle, Void arg) {
    super.visit(nle, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies string literals in a java method and sums them up to the total number of
   * literals
   *
   * @param StringLiteralExpr
   * @param Void
   */
  @Override
  public void visit(StringLiteralExpr sle, Void arg) {
    super.visit(sle, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies text block literals in a java method and sums them up to the total
   * number of literals
   *
   * @param TextBlockLiteralExpr
   * @param Void
   */
  @Override
  public void visit(TextBlockLiteralExpr tble, Void arg) {
    super.visit(tble, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies double literals in a java method and sums them up to the total number of
   * literals
   *
   * @param DoubleLiteralExpr
   * @param Void
   */
  @Override
  public void visit(DoubleLiteralExpr dle, Void arg) {
    super.visit(dle, arg);
    features.incrementNumOfLiterals();
  }

  /**
   * This method identifies block comments in a java method and sums them up to the total number of
   * comments
   *
   * @param BlockComment
   * @param Void
   */
  @Override
  public void visit(BlockComment comm, Void arg) {
    super.visit(comm, arg);
    features.incrementNumOfComments();
  } 

  /**
   * This method identifies block comments in a java method and sums them up to the total number of
   * comments
   *
   * @param JavadocComment
   * @param Void
   */
  @Override
  public void visit(JavadocComment comm, Void arg) {
    super.visit(comm, arg);
    features.incrementNumOfComments();
  } 
  
  /**
   * This method identifies block comments in a java method and sums them up to the total number of
   * comments
   *
   * @param LineComment
   * @param Void
   */
  @Override
  public void visit(LineComment comm, Void arg) {
    super.visit(comm, arg);
    features.incrementNumOfComments();
  } 

  /**
   * This method is to get the computed features After one/more visit method is/are called, the
   * features will be updated and then use this method to get the updated features
   *
   * @return features
   */
  public Features getFeatures() {
    return features;
  }
}

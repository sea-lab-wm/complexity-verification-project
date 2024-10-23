package edu.wm.sealab.featureextraction;

import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.expr.AssignExpr;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.BinaryExpr.Operator;
import com.github.javaparser.ast.expr.BooleanLiteralExpr;
import com.github.javaparser.ast.expr.CharLiteralExpr;
import com.github.javaparser.ast.expr.DoubleLiteralExpr;
import com.github.javaparser.ast.expr.IntegerLiteralExpr;
import com.github.javaparser.ast.expr.LongLiteralExpr;
import com.github.javaparser.ast.expr.NullLiteralExpr;
import com.github.javaparser.ast.expr.SimpleName;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.expr.TextBlockLiteralExpr;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.stmt.AssertStmt;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.BreakStmt;
import com.github.javaparser.ast.stmt.CatchClause;
import com.github.javaparser.ast.stmt.ContinueStmt;
import com.github.javaparser.ast.stmt.DoStmt;
import com.github.javaparser.ast.stmt.EmptyStmt;
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
import com.github.javaparser.ast.type.Type;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * This class to extract features from a java file Input : Java file with a Single Method (Note: if
 * want to use all the features) Output: Features
 */
public class FeatureVisitor extends VoidVisitorAdapter<Void> {

  private Features features = new Features();

  /**
   * This method creates a map of line number and count of given feature value
   */
  private Map<String, Double> constructLineNumberFeatureMap (Map<String, Double> map, String lineNumber) {
    if (map.containsKey(lineNumber)) {
      double numOfValues = map.get(lineNumber);
      map.put(lineNumber, numOfValues + 1.0);
    } else {
      map.put(lineNumber, 1.0);
    }
    return map;
  }

  /**
   * This method to compute #parameters of a java method
   */
  @Override
  public void visit(MethodDeclaration md, Void arg) {
    super.visit(md, arg);
    features.setNumOfParameters(md.getParameters().size());
  
    List<String> elements = new ArrayList<>();
    if (md.getParameters().size() > 0) {
        elements.addAll(Arrays.asList(md.getParameters().toString().split(",")));
    }
    if (md.getThrownExceptions().size() > 0) {
        elements.addAll(Arrays.asList(md.getThrownExceptions().toString().split(",")));
    }

    String lineNumber = md.getRange().get().begin.line + "";
    List<String> param_return_throws_List = features.getLineNumber_param_return_throws_Map().getOrDefault(lineNumber, new ArrayList<>());
    param_return_throws_List.addAll(elements);
    features.getLineNumber_param_return_throws_Map().put(lineNumber, param_return_throws_List);


    // check whether there are any class names inside a method.
    // This inclues both method sinature and method body
    for (Type type: md.findAll(Type.class)){
      features.getClassNames().add(type.asString());
    }
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
    // print the line number of the if statement and statement type
    // System.out.println("Line Number: " + ifStmt.getRange().get().begin.line + " Statement Type: " + ifStmt.getClass().getSimpleName());
    

    String lineNumber = ifStmt.getRange().get().begin.line+"";
    features.setLineConditionalMap(constructLineNumberFeatureMap(features.getLineConditionalMap(), lineNumber));    
  }

  /**
   * This method to compute # switch statements of a java method (not entries)
   */
  @Override
  public void visit(SwitchStmt swStmt, Void arg) {
    super.visit(swStmt, arg);
    features.incrementNumOfConditionals();
    features.incrementNumOfStatements();

    // print the line number of the statement and statement type
    // System.out.println("Line Number: " + swStmt.getRange().get().begin.line + " Statement Type: " + swStmt.getClass().getSimpleName());
    

    String lineNumber = swStmt.getRange().get().begin.line+"";
    features.setLineConditionalMap(constructLineNumberFeatureMap(features.getLineConditionalMap(), lineNumber));
  }

  /**
   * This method to compute # for loops of a java method
   */
  @Override
  public void visit(ForStmt forStmt, Void arg) {
    super.visit(forStmt, arg);
    features.incrementNumOfLoops();
    features.incrementNumOfStatements();

    // print the line number of the statement and statement type
    // System.out.println("Line Number: " + forStmt.getRange().get().begin.line + " Statement Type: " + forStmt.getClass().getSimpleName());
    

    String lineNumber = forStmt.getRange().get().begin.line+"";
    features.setLineLoopMap(constructLineNumberFeatureMap(features.getLineLoopMap(), lineNumber));
  }

  /**
   * This method to compute # while loops of a java method
   */
  @Override
  public void visit(WhileStmt whileStmt, Void arg) {
    super.visit(whileStmt, arg);
    features.incrementNumOfLoops();
    features.incrementNumOfStatements();

    // print the line number of the statement and statement type
    // System.out.println("Line Number: " + whileStmt.getRange().get().begin.line + " Statement Type: " + whileStmt.getClass().getSimpleName());

    String lineNumber = whileStmt.getRange().get().begin.line+"";
    features.setLineLoopMap(constructLineNumberFeatureMap(features.getLineLoopMap(), lineNumber));
  }

  /**
   * This method to compute # for each loops of a java method
   */
  @Override
  public void visit(ForEachStmt forEachStmt, Void arg) {
    super.visit(forEachStmt, arg);
    features.incrementNumOfLoops();
    features.incrementNumOfStatements();

    // print the line number of the statement and statement type
    // System.out.println("Line Number: " + forEachStmt.getRange().get().begin.line + " Statement Type: " + forEachStmt.getClass().getSimpleName());

    String lineNumber = forEachStmt.getRange().get().begin.line+"";
    features.setLineLoopMap(constructLineNumberFeatureMap(features.getLineLoopMap(), lineNumber));
  }

  /**
   * This method computes # assignment expressions in a java method
   * Does not include declaration statements
   */
  @Override
  public void visit(AssignExpr assignExpr, Void arg) {
    super.visit(assignExpr, arg);
    features.setAssignExprs(features.getAssignExprs() + 1);
    String lineNumber = assignExpr.getRange().get().begin.line+"";
    
    features.setLineAssignmentExpressionMap(constructLineNumberFeatureMap(features.getLineAssignmentExpressionMap(), lineNumber));
  }

  /*
   * This method computes # assignment expressions in a java method
   * Includes declaration statements
   */
  @Override
  public void visit(VariableDeclarationExpr vdExpr, Void arg) {
    super.visit(vdExpr, arg);
    features.setAssignExprs(features.getAssignExprs() + 1);
    String lineNumber = vdExpr.getRange().get().begin.line+"";
    
    features.setLineAssignmentExpressionMap(constructLineNumberFeatureMap(features.getLineAssignmentExpressionMap(), lineNumber));
  }

  /**
   * This method computes # comparisons and arithmetic operators in a java method
   */
  @Override
  public void visit(BinaryExpr n, Void arg) {
    super.visit(n, arg);
    Operator operator = n.getOperator();
    String lineNumber = n.getRange().get().begin.line+"";
    if (operator == Operator.EQUALS
        || operator == Operator.NOT_EQUALS
        || operator == Operator.LESS
        || operator == Operator.LESS_EQUALS
        || operator == Operator.GREATER
        || operator == Operator.GREATER_EQUALS) {
      features.setComparisons(features.getComparisons() + 1);

      features.setLineComparisonMap(constructLineNumberFeatureMap(features.getLineComparisonMap(), lineNumber));
      
      features.setLineOperatorMap(constructLineNumberFeatureMap(features.getLineOperatorMap(), lineNumber));
      
      // print line number and operator
      // System.out.println("Line Number: " + n.getRange().get().begin.line + " Operator: " + operator.asString());

      features.getOperators().add(operator.asString()); // add operators

    } else if (operator == Operator.PLUS
        || operator == Operator.MINUS
        || operator == Operator.MULTIPLY
        || operator == Operator.DIVIDE
        || operator == Operator.REMAINDER) {
      features.incrementNumOfArithmeticOperators();

      features.setLineOperatorMap(constructLineNumberFeatureMap(features.getLineOperatorMap(), lineNumber));

      features.getOperators().add(operator.asString()); // add operators

      // print line number and operator
      // System.out.println("Line Number: " + n.getRange().get().begin.line + " Operator: " + operator.asString());

    } else if (operator == Operator.AND
        || operator == Operator.OR
        || operator == Operator.BINARY_AND
        || operator == Operator.BINARY_OR
        || operator == Operator.XOR
        || operator == Operator.LEFT_SHIFT 
        || operator == Operator.SIGNED_RIGHT_SHIFT 
        || operator == Operator.UNSIGNED_RIGHT_SHIFT) {
      features.incrementNumOfLogicalOperators();

      features.setLineOperatorMap(constructLineNumberFeatureMap(features.getLineOperatorMap(), lineNumber));

      // print line number and operator
      // System.out.println("Line Number: " + n.getRange().get().begin.line + " Operator: " + operator.asString());


      features.getOperators().add(operator.asString()); // add operators
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
    features.getOperands().add(ble.toString()); // add operands
  }

  /**
   * This method identifies char literals in a java method and sums them up to the total number of
   * literals
   */
  @Override
  public void visit(CharLiteralExpr cle, Void arg) {
    super.visit(cle, arg);
    features.incrementNumOfLiterals();
    features.getOperands().add(cle.getValue()); // add operands
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
    features.getOperands().add(ile.getValue()); // add operands
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
    features.getOperands().add(lle.getValue()); // add operands
  }

  /**
   * This method identifies null literals in a java method and sums them up to the total number of
   * literals
   */
  @Override
  public void visit(NullLiteralExpr nle, Void arg) {
    super.visit(nle, arg);
    features.incrementNumOfLiterals();
    features.getOperands().add(nle.toString()); // add operands
  }

  /**
   * This method identifies string literals in a java method and sums them up to the total number of
   * literals
   */
  @Override
  public void visit(StringLiteralExpr sle, Void arg) {
    super.visit(sle, arg);
    features.incrementNumOfLiterals();
    features.getOperands().add(sle.getValue()); // add operands
  }

  /**
   * This method identifies text block literals in a java method and sums them up to the total
   * number of literals
   */
  @Override
  public void visit(TextBlockLiteralExpr tble, Void arg) {
    super.visit(tble, arg);
    features.incrementNumOfLiterals();
    features.getOperands().add(tble.getValue()); // add operands
  }

  /**
   * This method identifies double literals in a java method and sums them up to the total number of
   * literals
   */
  @Override
  public void visit(DoubleLiteralExpr dle, Void arg) {
    super.visit(dle, arg);
    features.incrementNumOfLiterals();
    features.getOperands().add(dle.getValue()); // add operands
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

    for (Comment comment : cu.getAllContainedComments()) {
      String lineNumber = comment.getRange().get().begin.line+"";
      features.setLineCommentMap(constructLineNumberFeatureMap(features.getLineCommentMap(), lineNumber));
    
      // add comments
      features.getComments().add(comment.getContent());

      // // print line number and comment
      System.out.println("Line Number: " + comment.getRange().get().begin.line + " Comment: " + comment.getContent());
    }
  }
  

  /**
   * This method identifies identifiers in a java method and sums them up to the total number of
   * operands
   */
  @Override
  public void visit(SimpleName sn, Void arg) {
    super.visit(sn, arg);

    // if identifier contain ds_[number]_snip_[number] then check regex and ignore
    // this will ignore the identifiers that are generated by the snippet splitter
    if(Pattern.compile("^ds_[a-zA-Z0-9]+[$_]").matcher(sn.getIdentifier()).find()){
      return;
    }
    

    features.incrementNumOfIdentifiers();
    features.getOperands().add(sn.getIdentifier()); // add operands
    
    String lineNumber = sn.getRange().get().begin.line+"";
    if(features.getLineNumber_Identifier_Map().containsKey(lineNumber)){
      List<String> identifierList = features.getLineNumber_Identifier_Map().get(lineNumber);
      
      identifierList.add(sn.getIdentifier());
      features.getLineNumber_Identifier_Map().put(lineNumber,identifierList);

      // print line number and identifier
      // System.out.println("Line Number: " + sn.getRange().get().begin.line + " Identifier: " + sn.getIdentifier());
    
    }else{
      List<String> identifierList = new ArrayList<String>();
      identifierList.add(sn.getIdentifier());
      features.getLineNumber_Identifier_Map().put(lineNumber,identifierList);

      // print line number and identifier
      // System.out.println("Line Number: " + sn.getRange().get().begin.line + " Identifier: " + sn.getIdentifier());
    }
  }

  /**
   * This method identifies Assert Statements in a java method to sum up the total number of
   * all statements
   */
  @Override
  public void visit(AssertStmt ast, Void arg) {
    super.visit(ast, arg);
    features.incrementNumOfStatements();

    // print line number and statement type
    // System.out.println("Line Number: " + ast.getRange().get().begin.line + " Statement Type: " + ast.getClass().getSimpleName());
  }

  /**
   * This method identifies Block Statements in a java method to sum up the total number of 
   * all statements. It also counts the number of nested blocks.
   */
  @Override
  public void visit(BlockStmt bst, Void arg) {
    super.visit(bst, arg);
    
    // counting nested blocks
    for (Node node : bst.getChildNodes()) {
      
      // Found {} block
      if (node instanceof BlockStmt) {
        features.incrementNumOfNestedBlocks();
      }

      if (node instanceof ForStmt) {
        features.incrementNumOfNestedBlocks();
      }
      if (node instanceof ForEachStmt) {
        features.incrementNumOfNestedBlocks();
      }
      if (node instanceof WhileStmt) {
        features.incrementNumOfNestedBlocks();
      }
      // found do {} while() block
      if (node instanceof DoStmt) {
        features.incrementNumOfNestedBlocks();
      }

      if (node instanceof IfStmt) {
        // Found else statement
        if (((IfStmt) node).getElseStmt().isPresent()) {
          features.incrementNumOfNestedBlocks();
        }
        // Found else if statement
        if (((IfStmt) node).getElseStmt().isPresent() && ((IfStmt) node).getElseStmt().get() instanceof IfStmt) {
          features.incrementNumOfNestedBlocks();
        }
        features.incrementNumOfNestedBlocks();
      }
      if (node instanceof SwitchStmt) {
        features.incrementNumOfNestedBlocks();
      }
      
      if (node instanceof TryStmt) {
        // Found finally block
        if (((TryStmt) node).getFinallyBlock().isPresent()) {
          features.incrementNumOfNestedBlocks();
        }
        // Found catch block
        if (((TryStmt) node).getCatchClauses().size() > 0) {
          features.incrementNumOfNestedBlocks();
        }
        // Found try block
        features.incrementNumOfNestedBlocks();
      }

      // found synchronized statement
      if (node instanceof SynchronizedStmt) {
        features.incrementNumOfNestedBlocks();
      }

      // increment number of statements
      features.incrementNumOfStatements();
    }
  }

  /**
   * This method identifies Break Statements in a java method to sum up the total number of 
   * all statements
   */
  @Override
  public void visit(BreakStmt brst, Void arg) {
    super.visit(brst, arg);
    features.incrementNumOfStatements();

    // print line number and statement type
    // System.out.println("Line Number: " + brst.getRange().get().begin.line + " Statement Type: " + brst.getClass().getSimpleName());
  }

  /**
   * This method identifies Catch Clauses in a java method to sum up the total number of 
   * all statements
   */
  // @Override
  // public void visit(CatchClause cc, Void arg) {
  //   super.visit(cc, arg);
  //   features.incrementNumOfStatements();

  //   // print line number and statement type
  //   System.out.println("Line Number: " + cc.getRange().get().begin.line + " Statement Type: " + cc.getClass().getSimpleName());
  // }

  /**
   * This method identifies Continue Statements in a java method to sum up the total number of 
   * all statements
   */
  @Override
  public void visit(ContinueStmt cs, Void arg) {
    super.visit(cs, arg);
    features.incrementNumOfStatements();

    // print line number and statement type
    // System.out.println("Line Number: " + cs.getRange().get().begin.line + " Statement Type: " + cs.getClass().getSimpleName());
  }

  /**
   * This method identifies Do Statements in a java method to sum up the total number of 
   * all statements
   */
  @Override
  public void visit(DoStmt ds, Void arg) {
    super.visit(ds, arg);
    features.incrementNumOfStatements();

    // print line number and statement type
    // System.out.println("Line Number: " + ds.getRange().get().begin.line + " Statement Type: " + ds.getClass().getSimpleName());
  }

  /**
   * This method identifies Empty Statements in a java method to sum up the total number of 
   * all statements.
   * eg. ;
   */
  @Override
  public void visit(EmptyStmt es, Void arg) {
    super.visit(es, arg);
    features.incrementNumOfStatements();

    // print line number and statement type
    // System.out.println("Line Number: " + es.getRange().get().begin.line + " Statement Type: " + es.getClass().getSimpleName());
  }

  /**
   * This method identifies Explicit Constructor Invocation Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(ExplicitConstructorInvocationStmt ecis, Void arg) {
    super.visit(ecis, arg);
    features.incrementNumOfStatements();

    // print line number and statement type
    // System.out.println("Line Number: " + ecis.getRange().get().begin.line + " Statement Type: " + ecis.getClass().getSimpleName());
  }

  /**
   * This method identifies Expression Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(ExpressionStmt es, Void arg) {
    super.visit(es, arg);
    features.incrementNumOfStatements();

    // print line number and statement type
    // System.out.println("Line Number: " + es.getRange().get().begin.line + " Statement Type: " + es.getClass().getSimpleName());
  }

  /**
   * This method identifies Labeled Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(LabeledStmt ls, Void arg) {
    super.visit(ls, arg);
    features.incrementNumOfStatements();

    // print line number and statement type
    // System.out.println("Line Number: " + ls.getRange().get().begin.line + " Statement Type: " + ls.getClass().getSimpleName());
  }

  /**
   * This method identifies Local Class Declaration Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(LocalClassDeclarationStmt lcds, Void arg) {
    super.visit(lcds, arg);
    features.incrementNumOfStatements();

    // print line number and statement type
    // System.out.println("Line Number: " + lcds.getRange().get().begin.line + " Statement Type: " + lcds.getClass().getSimpleName());
  }

  /**
   * This method identifies Local Record Declaration Statements in a java method to sum up the total number of 
   * all statements. // can't use since introduced in Java 14
   */
  // @Override
  // public void visit(LocalRecordDeclarationStmt lrds, Void arg) {
  //   super.visit(lrds, arg);
  //   features.incrementNumOfStatements();

  //   // print line number and statement type
  //   // System.out.println("Line Number: " + lrds.getRange().get().begin.line + " Statement Type: " + lrds.getClass().getSimpleName());
  // }

  /**
   * This method identifies Return Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(ReturnStmt rs, Void arg) {
    super.visit(rs, arg);
    features.incrementNumOfStatements();

    // print line number and statement type
    // System.out.println("Line Number: " + rs.getRange().get().begin.line + " Statement Type: " + rs.getClass().getSimpleName());

    String lineNumber = rs.getRange().get().begin.line+"";
    if(features.getLineNumber_param_return_throws_Map().containsKey(lineNumber)){
      List<String> param_return_throws_List = features.getLineNumber_param_return_throws_Map().get(lineNumber);
      param_return_throws_List.add(rs.getExpression().toString());
      features.getLineNumber_param_return_throws_Map().put(lineNumber,param_return_throws_List);
    }else{
      List<String> param_return_throws_List = new ArrayList<String>();
      param_return_throws_List.add(rs.getExpression().toString());
      features.getLineNumber_param_return_throws_Map().put(lineNumber,param_return_throws_List);
    }

  }

  /**
   * This method identifies Synchronized Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(SynchronizedStmt ss, Void arg) {
    super.visit(ss, arg);
    features.incrementNumOfStatements();

    // print line number and statement type
    // System.out.println("Line Number: " + ss.getRange().get().begin.line + " Statement Type: " + ss.getClass().getSimpleName());
  }

  /**
   * This method identifies Throw Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(ThrowStmt ts, Void arg) {
    super.visit(ts, arg);
    features.incrementNumOfStatements();

    // print line number and statement type
    // System.out.println("Line Number: " + ts.getRange().get().begin.line + " Statement Type: " + ts.getClass().getSimpleName());

    String lineNumber = ts.getRange().get().begin.line+"";
    if(features.getLineNumber_param_return_throws_Map().containsKey(lineNumber)){
      List<String> param_return_throws_List = features.getLineNumber_param_return_throws_Map().get(lineNumber);
      param_return_throws_List.add(ts.getExpression().toString());
      features.getLineNumber_param_return_throws_Map().put(lineNumber,param_return_throws_List);
    }else{
      List<String> param_return_throws_List = new ArrayList<String>();
      param_return_throws_List.add(ts.getExpression().toString());
      features.getLineNumber_param_return_throws_Map().put(lineNumber,param_return_throws_List);
    }

  }

  /**
   * This method identifies Try Statements in a java method to sum up the total number of 
   * all statements.
   */
  @Override
  public void visit(TryStmt ts, Void arg) {
    super.visit(ts, arg);
    features.incrementNumOfStatements();

    // print line number and statement type
    // System.out.println("Line Number: " + ts.getRange().get().begin.line + " Statement Type: " + ts.getClass().getSimpleName());
  }

  /**
   * This method identifies Yield Statements in a java method to sum up the total number of 
   * all statements. // Can't use since introduced in Java 14
   */
  // @Override
  // public void visit(YieldStmt ys, Void arg) {
  //   super.visit(ys, arg);
  //   features.incrementNumOfStatements();

  //   // print line number and statement type
  //   System.out.println("Line Number: " + ys.getRange().get().begin.line + " Statement Type: " + ys.getClass().getSimpleName());
  // }

  /**
   * This method is to get the computed features After one/more visit method is/are called, the
   * features will be updated and then use this method to get the updated features
   */
  public Features getFeatures() {
    return features;
  }
}

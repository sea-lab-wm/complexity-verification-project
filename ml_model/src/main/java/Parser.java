import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.visitor.VoidVisitor;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.CompilationUnit;

import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.expr.BooleanLiteralExpr;
import com.github.javaparser.ast.expr.CharLiteralExpr;
import com.github.javaparser.ast.expr.DoubleLiteralExpr;
import com.github.javaparser.ast.expr.IntegerLiteralExpr;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.List;

public class Parser {
    
    private static final String FILE_PATH = "ml_model/src/main/java/ReversePolishNotation.java";

    private static CompilationUnit cu;

    public static void main(String[] args) throws Exception {
        /*
        cu = StaticJavaParser.parse(new FileInputStream(FILE_PATH));

        List<Integer> methodNames = new ArrayList<>();
        VoidVisitor<List<Integer>> methodNameCollector = new MethodNameCollector();
        methodNameCollector.visit(cu, methodNames);
        methodNames.forEach(n -> System.out.println("Method Name Collected at Line: " + n));

        List<String> ifs = new ArrayList<>();
        VoidVisitor<List<String>> ifsCollector = new IfsCollector();
        ifsCollector.visit(cu, ifs);
        ifs.forEach(n -> System.out.println("Conditional Collected: " + n));

        List<String> ble = new ArrayList<>();
        VoidVisitor<List<String>> bleCollector = new BooleanLiteralExprCollector();
        bleCollector.visit(cu, ble);
        ble.forEach(n -> System.out.println("Conditional Collected: " + n));

        List<String> cle = new ArrayList<>();
        VoidVisitor<List<String>> cleCollector = new CharLiteralExprCollector();
        cleCollector.visit(cu, cle);
        cle.forEach(n -> System.out.println("Conditional Collected: " + n));

        List<Double> dle = new ArrayList<>();
        VoidVisitor<List<Double>> dleCollector = new DoubleLiteralExprCollector();
        dleCollector.visit(cu, dle);
        dle.forEach(n -> System.out.println("Conditional Collected: " + n));

        List<Integer> ile = new ArrayList<>();
        VoidVisitor<List<Integer>> ileCollector = new IntegerLiteralExprCollector();
        ileCollector.visit(cu, ile);
        ile.forEach(n -> System.out.println("Conditional Collected: " + n));
        */

        ParseFile(FILE_PATH);
    }

    /*
     * Parses a given file. Splits the file up by block statements (defined as anything between {}).
     * Determines which block statements represent a snippet (can be classes or methods). -> Assumes each snippet contains a comment identifying it directly above the method or class declaration.
     * Parses each snippet block individually for all features.
     * 
     * @return not sure
     */
    private static void ParseFile(String file_path) throws Exception {
        cu = StaticJavaParser.parse(new FileInputStream("simple-datasets/src/main/java/cog_complexity_validation_datasets/One/Tasks.java"));
        //cu = StaticJavaParser.parse(new FileInputStream("ml_model/src/main/java/Test.java"));
        //cu = StaticJavaParser.parse(new FileInputStream(file_path));

        List<MethodDeclaration> methodNames = new ArrayList<>();
        VoidVisitor<List<MethodDeclaration>> methodNameCollector = new MethodNameCollector();
        methodNameCollector.visit(cu, methodNames);
        //methodNames.forEach(n -> System.out.println("Method Name Collected at Line: " + n));

        for (MethodDeclaration md : methodNames) {
            //if (md.getBody().toString())
            CompilationUnit cu_md = StaticJavaParser.parse(md.getBody().toString());
            System.out.println(md.getBody().toString());

            List<String> ifs = new ArrayList<>();
            VoidVisitor<List<String>> ifsCollector = new IfsCollector();
            ifsCollector.visit(cu_md, ifs);
            ifs.forEach(n -> System.out.println("Conditional Collected: " + n));
        }
    }

    /*
     * Features
     */

    //
    private static class MethodNameCollector extends VoidVisitorAdapter<List<MethodDeclaration>> {

        @Override
        public void visit(MethodDeclaration md, List<MethodDeclaration> collector) {
            super.visit(md, collector);

            //int startLineNum = md.getRange().get().begin.line;
            //collector.add(startLineNum);

            if (md.getComment().toString().contains("SNIPPET_STARTS"))
                collector.add(md);
        }
    }

    private static class IfsCollector extends VoidVisitorAdapter<List<String>> {

        @Override
        public void visit(IfStmt ifs, List<String> collector) {
            super.visit(ifs, collector);
            collector.add(ifs.getCondition().toString());
        }
    }

    private static class BooleanLiteralExprCollector extends VoidVisitorAdapter<List<String>> {

        @Override
        public void visit(BooleanLiteralExpr ble, List<String> collector) {
            super.visit(ble, collector);
            collector.add(Boolean.toString(ble.getValue()));
        }
    }

    private static class CharLiteralExprCollector extends VoidVisitorAdapter<List<String>> {

        @Override
        public void visit(CharLiteralExpr cle, List<String> collector) {
            super.visit(cle, collector);
            collector.add(String.valueOf(cle.asChar()));
        }
    }

    private static class DoubleLiteralExprCollector extends VoidVisitorAdapter<List<Double>> {

        @Override
        public void visit(DoubleLiteralExpr dle, List<Double> collector) {
            super.visit(dle, collector);
            collector.add(dle.asDouble());
        }
    }

    private static class IntegerLiteralExprCollector extends VoidVisitorAdapter<List<Integer>> {

        @Override
        public void visit(IntegerLiteralExpr ile, List<Integer> collector) {
            super.visit(ile, collector);
            collector.add(ile.asNumber().intValue());
        }
    }
}

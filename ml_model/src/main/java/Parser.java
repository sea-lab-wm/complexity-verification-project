import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.visitor.VoidVisitor;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.CompilationUnit;

import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.ConditionalExpr;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.List;

public class Parser {
    
    private static final String FILE_PATH = "src/main/java/ReversePolishNotation.java";

    public static void main(String[] args) throws Exception {
        CompilationUnit cu = StaticJavaParser.parse(new FileInputStream(FILE_PATH));

        List<String> methodNames = new ArrayList<>();
        VoidVisitor<List<String>> methodNameCollector = new MethodNameCollector();
        methodNameCollector.visit(cu, methodNames);
        methodNames.forEach(n -> System.out.println("Method Name Collected: " + n));

        List<String> conditionals = new ArrayList<>();
        VoidVisitor<List<String>> conditionalCollector = new ConditionalCollector();
        System.out.println("TEST2");
        conditionalCollector.visit(cu, conditionals);
        System.out.println("TEST3");
        conditionals.forEach(n -> System.out.println("Conditional Collected: " + n));
    }

    private static class MethodNameCollector extends VoidVisitorAdapter<List<String>> {

        @Override
        public void visit(MethodDeclaration md, List<String> collector) {
            super.visit(md, collector);
            collector.add(md.getNameAsString());
        }
    }

    private static class ConditionalCollector extends VoidVisitorAdapter<List<String>> {

        @Override
        public void visit(ConditionalExpr ce, List<String> collector) {
            super.visit(ce, collector);
            collector.add(ce.getCondition().toString());
            System.out.println("TEST");
        }
    }
}

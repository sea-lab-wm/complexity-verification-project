import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.visitor.VoidVisitor;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.CompilationUnit;

import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.IfStmt;

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

        List<String> ifs = new ArrayList<>();
        VoidVisitor<List<String>> ifsCollector = new IfsCollector();
        ifsCollector.visit(cu, ifs);
        ifs.forEach(n -> System.out.println("Conditional Collected: " + n));
    }

    private static class MethodNameCollector extends VoidVisitorAdapter<List<String>> {

        @Override
        public void visit(MethodDeclaration md, List<String> collector) {
            super.visit(md, collector);
            collector.add(md.getNameAsString());
        }
    }

    private static class IfsCollector extends VoidVisitorAdapter<List<String>> {

        @Override
        public void visit(IfStmt ifs, List<String> collector) {
            super.visit(ifs, collector);
            collector.add(ifs.getCondition().toString());
        }
    }
}

italian_keywords = list()
italian_keywords.append("abstract")
italian_keywords.append("continue")
italian_keywords.append("for")
italian_keywords.append("new")
italian_keywords.append("switch")
italian_keywords.append("assert")
italian_keywords.append("default")
italian_keywords.append("if")
italian_keywords.append("package")
italian_keywords.append("synchronized")
italian_keywords.append("boolean")
italian_keywords.append("do")
italian_keywords.append("goto")
italian_keywords.append("private")
italian_keywords.append("this")
italian_keywords.append("break")
italian_keywords.append("double")
italian_keywords.append("implements")
italian_keywords.append("protected")
italian_keywords.append("throw")
italian_keywords.append("byte")
italian_keywords.append("else")
italian_keywords.append("import")
italian_keywords.append("public")
italian_keywords.append("throws")
italian_keywords.append("case")
italian_keywords.append("enum")
italian_keywords.append("instanceof")
italian_keywords.append("return")
italian_keywords.append("transient")
italian_keywords.append("catch")
italian_keywords.append("extends")
italian_keywords.append("int")
italian_keywords.append("short")
italian_keywords.append("try")
italian_keywords.append("char")
italian_keywords.append("final")
italian_keywords.append("interface")
italian_keywords.append("static")
italian_keywords.append("void")
italian_keywords.append("class")
italian_keywords.append("finally")
italian_keywords.append("long")
italian_keywords.append("strictfp")
italian_keywords.append("volatile")
italian_keywords.append("const")
italian_keywords.append("float")
italian_keywords.append("native")
italian_keywords.append("super")
italian_keywords.append("while")

my_keywords = {"abstract", 
  "assert", 
  "boolean", 
  "break", 
  "byte", 
  "case", 
  "catch", 
  "char", "class", "const", "continue",
  "default", "do", "double", "else", "enum", "extends",
  "final", "finally", "float", "for", "goto", "if",
  "implements", "import", "instanceof", "int", "interface",
  "long", "native", "new", "package", "private", "protected",
  "public", "return", "short", "static", "strictfp", "super",
  "switch", "synchronized", "this", "throw", "throws", "transient",
  "try", "void", "volatile", "while"}

my_keywords = list(my_keywords)
print (len(my_keywords))
print (len(italian_keywords))

## take the items only appear in both Italian list and my list
common_keywords = list(set(italian_keywords) & set(my_keywords))
## delete the common items from my list and Italian list and print them
for item in common_keywords:
    my_keywords.remove(item)
    italian_keywords.remove(item)
print("Italian keywords: ", italian_keywords)
print("My keywords: ", my_keywords)


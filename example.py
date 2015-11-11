from stat_parser import Parser, display_tree


parser = Parser()

[tree1, tree2] = parser.parse("John saw Mary with the telescope")

display_tree(tree1)
display_tree(tree2)

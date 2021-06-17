from Algorithm.Tree.decision_tree import decision_tree

def Get_Classifer (name = ''):
	switcher = {
		'decision_tree':decision_tree
	}
	res = switcher.get(name)
	if res == None:
		raise Exception('No Criterion Named: "{name}".'.format(name = name));
	return res
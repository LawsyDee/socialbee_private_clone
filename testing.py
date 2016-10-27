class CypherStatement(str):
	'''String subclass for writing clean cypher update statements
	'''

	def __init__(self,x):
		CypherStatement.name = x

	def say_hello(self):
		print("Hello "+CypherStatement.name)

def func_test(string,num):
	print((string+' ')*num)
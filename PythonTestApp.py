import tensorflow as tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Just disables the warning, doesn't enable AVX/FMA

a = tensorflow.constant([2])
b = tensorflow.constant([3])

c = tensorflow.add(a, b)

with tensorflow.Session() as session:
	result = session.run(c)
	print(result)
	

def test():
	graph = tensorflow.Graph()
	with graph.as_default():
	    a = tensorflow.constant([2])
	    b = tensorflow.constant([3])
	
	with tensorflow.Session(graph = graph) as session:
		c = tensorflow.add(a, b)	
		print(session.run(c))
	
	
	graph2 = tensorflow.Graph()
	with graph2.as_default():
	    Scalar = tensorflow.constant(2)
	    Vector = tensorflow.constant([5,6,2])
	    Matrix = tensorflow.constant([[1,2,3],[2,3,4],[3,4,5]])
	    Tensor = tensorflow.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )
	with tensorflow.Session(graph = graph2) as sess:
	    result = sess.run(Scalar)
	    print ("Scalar (1 entry):\n %s \n" % result)
	    result = sess.run(Vector)
	    print ("Vector (3 entries) :\n %s \n" % result)
	    result = sess.run(Matrix)
	    print ("Matrix (3x3 entries):\n %s \n" % result)
	    result = sess.run(Tensor)
	    print ("Tensor (3x3x3 entries) :\n %s \n" % result)
	return graph, graph2
test()
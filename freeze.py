# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:28:30 2018

@author: avarfolomeev
"""
sess = tf.Session()

netname = 'my2-net-3029'

saver = tf.train.import_meta_graph('/media/avarfolomeev/storage/Data/Segmentation/net/' 
                                    + netname + '.meta')
saver.restore(sess,'/media/avarfolomeev/storage/Data/Segmentation/net/'
                    + netname)

out_graph = '/media/D/DIZ/CityScapes/net/' + netname + '-frozen.pb'
out_name = 'layer3_up/BiasAdd'#

graph_def = tf.get_default_graph().as_graph_def()

output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            graph_def, # The graph_def is used to retrieve the nodes 
            [out_name] # The output node names are used to select the usefull nodes
        )


with tf.gfile.GFile(out_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

print("%d ops in the input graph." % len(graph_def.node))
print("%d ops in the final graph." % len(output_graph_def.node))


from IPython.display import Image
from sklearn import tree
import pydotplus 
from sklearn.tree import export_graphviz

def TreeShow(dtClass,irisDataSet):
    dot_data = export_graphviz(dtClass, out_file=None)\n",
    graph = pydotplus.graph_from_dot_data(dot_data)\n",
    graph.write_pdf(\"tree.pdf\")\n",
    dot_data = export_graphviz(dtClass, out_file=None,\n",
                               feature_names=['pack_type','time_tranf','flow_tranf',
                               'pack_change','contract','asso_pur','group_user'],   #对应特征的名字
                               class_names=['loss','not loss'],    #对应类别的名字
                               filled=True, rounded=True,
                               special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())
	
TreeShow(clf,df)
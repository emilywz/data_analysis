from IPython.display import Image
from sklearn import tree
import pydotplus 
from sklearn.tree import export_graphviz

#graphviz显示用户流失数据预测决策树
def TreeShow(dtClass,irisDataSet):
    dot_data = export_graphviz(dtClass, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(\"tree.pdf\")
    dot_data = export_graphviz(dtClass, out_file=None,
                               feature_names=['pack_price','time_tranf','flow_tranf',
                               'pack_change','contract','link_puchase','group_user'],   #对应特征的名字
                               class_names=['loss','not loss'],    #对应类别的名字
                               filled=True, 
							   rounded=True,
                               special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    #Image(graph.create_png())
	
TreeShow(clf,df)
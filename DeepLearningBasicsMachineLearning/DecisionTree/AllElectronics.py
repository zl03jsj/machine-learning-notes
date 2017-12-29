import csv
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer

cvs_filename = 'samples.csv'
# #convert excel to csv
# DecisionTree.xls_csv.excel_to_csv('./samples.xlsx', 'Sheet1', cvs_filename)

headers = []
rows = []
label_name = ''

# 最后一列为标签[label]值
with open(cvs_filename, 'r+t') as csvfile:
    reader = csv.DictReader(csvfile)
    field_count = len(reader.fieldnames)
    headers = reader.fieldnames[1:field_count-1]
    label_name = reader.fieldnames[field_count-1]
    rows = [row for row in reader]

feature_list = []
label_list = []

for row in rows:
    feature = {}
    label_list.append(row.get(label_name))

    for key in headers: feature[key] = row.get(key)
    feature_list.append(feature)

print(feature_list)
print(label_list)

vec = DictVectorizer()
dummy_x = vec.fit_transform(feature_list).toarray()
lab = LabelBinarizer()
dummy_y = lab.fit_transform(label_list)

print(dummy_x)
print(dummy_y)

# 使用信息熵作为判断决策树中节点是否选择某一属性作为决策分支的度量条件
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf.fit(dummy_x, dummy_y)

print('clf:', str(clf))

with open('allEleInformationGainOri.dot', 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

# to plot decision tree graph to pdf file,
# run next command in terminator:
# dot -Tpdf allEleInformationGainOri.dot -o output.pdf


new_row_x = dummy_x[0, :]
new_row_x[0] = 1
new_row_x[2] = 0

new_rows = []
new_rows.append(new_row_x)

is_buy = lab.classes_[clf.predict(new_rows)[0]]

print('预测结果: Class_buys_computer = ', is_buy)

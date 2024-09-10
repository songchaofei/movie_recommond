# coding=utf-8
from neo4j import GraphDatabase
import warnings
import os
import fileinput
warnings.filterwarnings('ignore')


driver = GraphDatabase.driver(uri='neo4j://192.168.126.128:7687', auth=('neo4j', 'sCF.521821+'))


def _load_data(path):
    """
    加载经过模型审核后的数据
    :param path:
    :return:
    """
    # 获取所有疾病对应的csv列表
    diseases_csv_list = os.listdir(path)
    # 将文件后缀名.csv去掉 提取出疾病的名称
    diseases_list = list(map(lambda x: x.split('.')[0], diseases_csv_list))

    # 将每一种疾病对应的症状放在症状列表中
    symptom_list = []
    for diseases_csv in diseases_csv_list:
        symptom = list(map(lambda x: x.strip(), fileinput.FileInput(os.path.join(path, diseases_csv), encoding='utf-8')))
        # 过滤掉所有长度异常的疾病症状
        symptom = list(filter(lambda x: 0< len(x) < 100, symptom))
        symptom_list.append(symptom)

    return dict(zip(diseases_list, symptom_list))


def write(path):
    """
    写入图数据库的函数
    :param path:
    :return:
    """
    # 导入数据称为字典类型  {疾病1:[症状1,症状2，....], 疾病2:[症状1，症状2，.....]}
    diseases_samptom_dict = _load_data(path)
    # 写入图数据库
    with driver.session() as session:
        """
        for key, value in diseases_samptom_dict.items():
            # 创建疾病名称的节点
            cypher = 'MERGE (a: Diseases{name:%r})'%key
            session.run(cypher)
            # 遍历创建症状节点
            for v in value:
                cypher = 'MERGE (b: Symptom{name:%r})'%v
                session.run(cypher)
                # 创建关系
                cypher = 'MATCH (a:Diseases{name: %r}) MATCH(b: Symptom{name: %r}) \
                 WITH a,b MERGE (a)-[r:dis_to_sym]-(b)'%(key, v)
                session.run(cypher)
        """

        # 在疾病节点上创建索引 (仅执行一次 重复执行会报错)
        cypher = 'CREATE INDEX FOR (a:Diseases) on (a.name)'
        session.run(cypher)
        # 在症状节点上创建索引
        cypher = 'CREATE INDEX FOR (b:Symptom) on (b.name)'
        session.run(cypher)


if __name__ == '__main__':
    path = '../doctor_data/structured/reviewed'
    write(path)
# coding=utf-8
from neo4j import GraphDatabase
import warnings
warnings.filterwarnings('ignore')


# NEO4J_CONFIG = {
#     'uri': 'neo4j://192.168.126.128:7687',
#     'port': 7687,
#     'user': 'neo4j',
#     'password': 'sCF.521821+'
# }


driver = GraphDatabase.driver(uri='neo4j://192.168.126.128:7687', auth=('neo4j', 'sCF.521821+'))


def _some_operation(tx, cat_name, mouse_name):
    tx.run('MERGE (a:Cat{name:$cat_name})'
           'MERGE (b:Mouse{name:$mouse_name})'
           'MERGE (a)-[r:And]-(b)',
           cat_name=cat_name, mouse_name=mouse_name)


with driver.session() as session:
    session.write_transaction(_some_operation, 'Tom', 'Jerry')
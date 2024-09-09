# coding=utf-8

REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379
}

# 导入redis驱动
import redis


# 创建一个redis连接池
pool = redis.ConnectionPool(**REDIS_CONFIG)
# 初始化一个活跃得连接对象
r = redis.StrictRedis(connection_pool=pool)
# uid 代表某个用户得唯一标识
uid = '8888'
# key是需要记录得数据描述
key = '该用户最后说的一句话:'.encode('utf-8')
value = '再见，董小姐'.encode('utf-8')
r.hset(uid, key, value)

# hget获取value值
res = r.hget(uid, key)
print(res)
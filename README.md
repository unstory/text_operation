# text_operation

* 文本操作
#### * 相似性度量算法
1. edit_distance(编辑距离)
算法思想：为了使两个字符串相等所插入、替换或删除的字符串数量
2. jaccard系数
算法思想：两个集合交集的相似程度

#### 关键词提取算法
1. tf-idf算法提取关键词
* 加载已有的文档数据集
* 加载停用词表
* 对数据集中的文档进行分词
* 根据停用词表，过滤干扰词
* 根据数据集训练算法,计算tf-idf值，取出关键词前10个

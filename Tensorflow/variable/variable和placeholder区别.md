variable必须初始化，主要用于存训练变量，如模型的权重，偏差。placeholder不用初始化，在session.run(xx, feed_dict)时指定，主要用于输入样本。

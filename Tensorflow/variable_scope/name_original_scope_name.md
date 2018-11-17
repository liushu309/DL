## sc.name与sc.original_scope_name的区别
  一般情况下, sc.name 与 sc.original_scope_name 返回的字符串相同，但是当相同的scope被创建的时候，sc.original_scope_name 会在 scope 参数后面加上_x(序号)，而 sc.name 则不会
### 示例
    import tensorflow as tf
    with tf.variable_scope('a') as a:
        print(a.name)
        print(a.original_name_scope)

    with tf.variable_scope('a') as b:
        print(b.name)
        print(b.original_name_scope)
    [out]:
    a
    a/
    a
    a_1/

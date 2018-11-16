## arg_scope
  给函数添加默认的参数，在使用些功能的函数定义前添加@slib.add_arg_scope，在函数调用前使用with slim.arg_scope([函数名]， 函数参数名 = 值， ...):

    import tensorflow as tf 
    import tensorflow.contrib.slim as slim

    @slim.add_arg_scope
    def test_fn(a, b, c):
        print(a)
        print(b)
        print(c)

    with slim.arg_scope([test_fn], b = 'liushu', c = 'mao'):
        test_fn(a = 'wo qu')

    # [out]:
    #     wo qu
    #     liushu
    #     mao
 

    
 

    import tensorflow as tf 
    import matplotlib.pyplot as plt 


    def preprocess_image(image_data):
        # 转换数据
        if image_data.dtype != tf.float32:
            image_data = tf.image.convert_image_dtype(image_data, tf.float32)
        # 缩放
        # process_img = tf.image.resize_images(image_data, [100, 100], method = 0)
        # 裁剪填充
        # process_img = tf.image.resize_image_with_crop_or_pad(image_data, 200, 1000)
        # 比例缩放
        # process_img = tf.image.central_crop(image_data, 0.5)
        # 上下翻转，也有随机random_flip_up_down
        # process_img = tf.image.flip_up_down(image_data)
        # 左右翻转, 也有随机random_flip_left_right
        # process_img = tf.image.flip_left_right(image_data)
        # 对角线翻转
        # process_img = tf.image.transpose_image(image_data)
        # 调整亮度
        # process_img = tf.image.adjust_brightness(image_data, -0.5)
        # process_img = tf.image.random_brightness(image_data, 1.0) # 在[-1.0, 1)范围内随机选取
        # 调整对比度
        # process_img = tf.image.adjust_contrast(image_data, 5)
        # process_img = tf.image.random_contrast(image_data, 1.0, 10.0) # 在[1.0, 10]范围内随机选取, 不能小于0
        # 调整色相
        # process_img = tf.image.adjust_hue(image_data, 0.1)
        # process_img = tf.image.random_hue(image_data, 0.5) # max_delta must be <= 0.5.
        # 调整饱和度
        # process_img = tf.image.adjust_saturation(image_data, -5)
        # process_img = tf.image.random_saturation(image_data, 0, 1)
        # 白化(标准化)
        # process_img = tf.image.per_image_standardization(image_data)
        # 在图像中画框 
        # image_data = tf.expand_dims(image_data, 0) # 因为认为输入为batch,所以将图像添加一维
        # boxes = tf.constant([[[0.1, 0.1, 0.5, 0.5]]]) # 输入的bounding boxes为三维，不是二维！形式如：[y_min, x_min, y_max, x_max]，即左上角下右下角
        # process_img = tf.image.draw_bounding_boxes(image_data, boxes)
        # process_img = process_img[0]
        # 随机截取图像（ROI）
        boxes = tf.constant([[[0., 0., 0.5, 0.5]]]) # 输入的bounding boxes为三维，不是二维！
        begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(image_data), boxes)
        process_img = tf.slice(image_data, begin, size)

        return process_img

    mat_file = tf.gfile.FastGFile("/home/liushu/Documents/Project/Python/SSD-Tensorflow-master/test/lenna.jpg", "rb")
    mat_tf = mat_file.read()
    mat_decode = tf.image.decode_jpeg(mat_tf)

    with tf.Session() as sess:
        mat_res = preprocess_image(mat_decode)
        mat_res = sess.run(mat_res)

        print(mat_res.dtype)
        print(mat_res.shape)
        plt.imshow(mat_res)
        plt.show()

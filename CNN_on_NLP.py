import tensorflow as tf
import numpy as np
import ReadCsv as read_utils
import utils
from tensorflow import graph_util

class CNN(object):
    def __init__(self):
        self.LENTH_MAX = 20 
        self.LOAD_SIZE = 4985   ## 由于语料原因，最多只能是4985
        self.VOCABULARY_SIZE = 10000
        self.EMBEDING_SIZE = 128
        self.BATCH_SIZE = 50
        self.KEEP_PROB = 0.7     
        self.LEARNING_RATE = 0.01
        self.CLASS_NUM = 3     ## '好评':0, '中评':1, '差评':2
        self.MAX_INTERATION = 20
        self.GUANLIAN = 5

    def interface(self, input, keep_prob):
        with tf.variable_scope("Layer_1"):         ##     定义命名域
            W1 = utils.weight_variable([5, 5, 1, 32], name="W1")  ## 构造变量W1
            b1 = utils.bias_variable([32], name="b1")
            conv_1 = utils.conv2d_basic(input, W1, b1)    ## 卷积
            relu_1 = tf.nn.relu(conv_1, name="relu_1")    ## 使用激活函数relu
            ## pool_1: size [BATCH_SIZE, 32, 64, 32]
            pool_1 = utils.avg_pool_2x2(relu_1)           ## 对激活函数输出进行平均卷积

        with tf.variable_scope("Layer_2"):
            W2 = utils.weight_variable([5, 5, 32, 256], name="W2")
            b2 = utils.bias_variable([256], name="b2")
            conv_2 = utils.conv2d_basic(pool_1, W2, b2)
            relu_2 = tf.nn.relu(conv_2, name='relu_2')
            ## pool_2: size [BATCH_SIZE, 16, 32, 256]
            pool_2 = utils.avg_pool_2x2(relu_2)
            # scope.reuse_variables()

        with tf.variable_scope("Layer_3"):
            W3 = utils.weight_variable([3, 3, 256, 384], name="W3")
            b3 = utils.bias_variable([384], name="b3")
            conv_3 = utils.conv2d_basic(pool_2, W3, b3)
            relu_3 = tf.nn.relu(conv_3, name="relu_3")
            ## pool_3: size [BATCH_SIZE, 8,16,384]
            pool_3 = utils.avg_pool_2x2(relu_3)

        with tf.variable_scope("Layer_4"):
            W4 = utils.weight_variable([3, 3, 384, 384], name="W4")
            b4 = utils.bias_variable([384], name="b4")
            conv_4 = utils.conv2d_basic(pool_3, W4, b4)
            relu_4 = tf.nn.relu(conv_4, name="relu_4")
            ## pool_4: size [BATCH_SIZE, 4,8, 384]
            pool_4 = utils.avg_pool_2x2(relu_4)

        with tf.variable_scope("Layer_5"):
            W5 = utils.weight_variable([3, 3, 384, 256], name="W5")
            b5 = utils.weight_variable([256], name="b5")
            conv_5 = utils.conv2d_basic(pool_4, W5, b5)
            relu_5 = tf.nn.relu(conv_5, name="relu_5")
            ## pool_5: size [BATCH_SIZE, 2, 4, 256]
            pool_5 = utils.avg_pool_2x2(relu_5)

        with tf.variable_scope("all_link"):   ### 全连接层
            W_fc1 = utils.weight_variable([1 * 4 * 256, 4096], name="W_fc1")
            b_fc1 = utils.bias_variable([4096], name="b_fc1")
            pool_5_flag = tf.reshape(pool_5, [-1, 1 * 4 * 256])
            ## h_fc1: size [BATCH_SIZE, 4096]
            h_fc1 = tf.nn.relu(tf.nn.xw_plus_b(pool_5_flag, W_fc1, b_fc1))
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            W_fc2 = utils.weight_variable([4096, self.CLASS_NUM], name="W_fc2")
            b_fc2 = utils.bias_variable([self.CLASS_NUM], name="b_fc2")

            h_fc2 = tf.nn.tanh(tf.nn.xw_plus_b(h_fc1_drop, W_fc2, b_fc2))
            # h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

            return h_fc2

    def interface_column(self, input, keep_prob):    ## 全行扫描（及卷积核单向移动）网络

        lenth = input.shape[2]     ## 提取输入矩阵的长度
        layers = []    ## 用于统计不同呢的卷积核卷积出来的结果，方便后来的合成
        guanlian = self.GUANLIAN-2    ## 关联长度，也就是指卷积核最多框住多少个词，最少2个，最大为：self.GUANLIAN
        with tf.variable_scope('Layer_1'):
            for i in range(2, self.GUANLIAN):
                name = 'conv_'+str(i)
                kernels = tf.Variable(tf.random_normal(shape = [i, self.EMBEDING_SIZE, 1, 1]))
                # print(i)
                # print(kernels.shape)
                bias = tf.constant(0.0, shape=[1])
                ## current.shape = [BATCH_SIZE, LENTH_MAX, 1, 1]
                ## 横向移动距离为lenth，即矩阵本身长度，保证不会左右平移，只会在列方向移动
                current = tf.nn.conv2d(input, kernels, strides = [1, 1, lenth, 1], padding='SAME', name = name)
                # 把计算出来的这一层保存起来，便于后面的激活函数预算与全连接
                layers.append(current)
            ## layer_1.shape = [BATCH_SIZE, LENTH_MAX, 1, GUANLIAN]
            layer_1 = tf.concat(layers, axis=3)
            relu = tf.nn.relu(layer_1, name='relu')

        with tf.variable_scope('aver_pooling'):
            size = relu.shape
            ## pool.shape = [BATCH_SIZE, 1, 1, GUANLIAN]
            pool = tf.nn.avg_pool(relu, ksize=[1, size[1], size[2], 1], strides=[1, size[1], size[2], 1], padding='SAME')

        with tf.variable_scope("all_link"):
            h_fc1_drop = tf.nn.dropout(pool, keep_prob)
            W_fc1 = utils.weight_variable([guanlian, self.CLASS_NUM], name="W_fc1")
            b_fc1 = utils.bias_variable([self.CLASS_NUM], name="b_fc1")
            pool_5_flag = tf.reshape(h_fc1_drop, [self.BATCH_SIZE, -1])
            # print(relu.shape)
            # print(W_fc1.shape)
            ## h_fc1: size [BATCH_SIZE, 4096]
            h_fc1 = tf.nn.relu(tf.nn.xw_plus_b(pool_5_flag, W_fc1, b_fc1))

            return h_fc1
    
    def CNNTrain(self, mode, BATCH_SIZE=None, text = None): 
        """
        Args:
            mode = "train" or "use"
        """
        if(BATCH_SIZE ==None) & (mode =="train"):
            BATCH_SIZE = self.BATCH_SIZE
        elif (BATCH_SIZE==None):
            BATCH_SIZE = 1
        # BATCH_SIZE = self.BATCH_SIZE

        text_input = tf.placeholder(
            dtype=tf.int32, shape=[BATCH_SIZE, self.LENTH_MAX], name="text_input")
        text_input_change = tf.reshape(text_input, [BATCH_SIZE, -1], name="Input")  ## 如果有3维向量的时候就需要reshape了，此处没有变化
        keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        label = tf.placeholder(dtype=tf.int64, shape=[BATCH_SIZE, ], name="label")

        embeding_var = tf.Variable(tf.random_uniform(
            shape=[self.VOCABULARY_SIZE, self.EMBEDING_SIZE]), dtype=tf.float32, name='embeding_var')

        ## batch_embeding: size [BATCH_SIZE, LENTH_MAX, EMBEDING_SIZE]
        batch_embeding = tf.nn.embedding_lookup(embeding_var, text_input_change)

        batch_embeding_normal = tf.reshape(
            batch_embeding, [-1, self.LENTH_MAX, self.EMBEDING_SIZE, 1])
        print(batch_embeding_normal)
        # output = self.interface(batch_embeding_normal, keep_prob)
        output = self.interface_column(batch_embeding_normal, keep_prob)
        # print(output)
        # print(label)

        with tf.variable_scope("loss"):
        	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            	logits=output, labels=label, name="loss"))


        with tf.variable_scope("accuracy"):
        # 	accuracy = tf.reduce_mean(
        #   	  tf.cast(tf.equal(tf.round(tf.sigmoid(output)), label), dtype=tf.float32))
        	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.sigmoid(output),1), label), dtype = tf.float32))

        result = tf.nn.softmax(output, name='result')
        # print(result)
        with tf.variable_scope("train_op"):
            train_op = tf.train.AdadeltaOptimizer(self.LEARNING_RATE).minimize(loss)
        saver = tf.train.Saver()

        if mode == 'train':    
            # print("output:")
            # print(output)
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
            summary_op = tf.summary.merge_all()

            accuracy_average = 0
            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())

                train_datareader = read_utils.ReadData(
                    self.LENTH_MAX, mode="train", load_size=self.LOAD_SIZE, vocabulary_size=self.VOCABULARY_SIZE)
                train_datareader.Train_Init()
                
                test_datareader = read_utils.ReadData(
                    self.LENTH_MAX, mode="test", load_size=self.LOAD_SIZE, vocabulary_size=self.VOCABULARY_SIZE)
                test_datareader.Train_Init()

                summary_writer = tf.summary.FileWriter("./logs", sess.graph)

                for itr in range(self.MAX_INTERATION):
                    x, y = train_datareader.NextBatch(BATCH_SIZE)
                    feed_dict = {text_input: x, label: y, keep_prob: self.KEEP_PROB}
                    sess.run(train_op, feed_dict)
                    # print(itr)

                    if itr % 10 == 0:
                        loss_train, accuracy_train, summary_str = sess.run(
                            [loss, accuracy, summary_op], feed_dict)

                        print("Step: %d , Train_loss: %g ,Train_accuracy: %g" %
                              (itr, loss_train, accuracy_train))   ####  每10次的训练后进行输出的语句
                        accuracy_average += accuracy_train
                        summary_writer.add_summary(summary_str, itr)

                    if itr % 100 == 0:
                        x_test, y_test = test_datareader.NextBatch(BATCH_SIZE)
                        loss_test, accuracy_test = sess.run([loss, accuracy], feed_dict={
                                                            text_input: x_test, label: y_test, keep_prob: 1})
                        print("Step: %d , loss_loss: %g ,loss_accuracy: %g" %
                              (itr, loss_test, accuracy_test))    ###   每100次训练后对测试集进行测试输出的语句

                saver.save(sess, "./log/CNN-model.ckpt")
                # saver2.save
                # tf.get_default_graph()返回当前会话的默认图
                # as_graph_def()返回一个图的序列化的GraphDef表示
                #               序列化的GraphDef可以导入至另一个图中(使用 import_graph_def())
                #               或者使用C++ Session API
                graph_def = tf.get_default_graph().as_graph_def()

                # graph_util模块的convert_variables_to_constants(
                #                     sess,              ----------------变量所在的会话
                #                     input_graph_def,      -------------持有需要保存的网络结构的GraphDef对象
                #                     output_node_names,    -------------需要保存的节点名称，注意命名域[scope_name/op_name]
                #                     variable_names_whitelist=None,   --要转换的变量名(在默认情况下，所有的变量都被转换)。
                #                     variable_names_blacklist=None   ---变量名的集合，省略转换为常量
                #                 )
                output_graph = graph_util.convert_variables_to_constants(sess, graph_def, ['result'])
                # output_graph = graph_util.remove_training_nodes(graph_def, ['all/result'])
                # output_graph = graph_util.extract_sub_graph(graph_def, ['all/result'])
                with tf.gfile.GFile("./log/combined_model.pb","wb") as f:
                    f.write(output_graph.SerializeToString())
            num = self.MAX_INTERATION/10
            accuracy_average = accuracy_average/num
            print("accuracy_average: %g "%accuracy_average)

        else:
            with tf.Session() as sess:

                saver.restore(sess, "./log/CNN-model.ckpt")
                train_datareader = read_utils.ReadData(
                    self.LENTH_MAX, mode="train", load_size=self.LOAD_SIZE, vocabulary_size=self.VOCABULARY_SIZE)
                x = train_datareader.Test(text)
                print(x)
                feed_dict = {text_input:x, keep_prob: 1 }
                print(result.eval(feed_dict))

    def test(self, text):
        """
        Args:
            text:需要进行预测情感分类的文本
        Return：
            result：文本分类的3中可能的概率，'好评':0, '中评':1, '差评':2 
        """
        model_filename = "./log/combined_model.pb"
        # 先是新建一个图
        graph = tf.Graph()
        with graph.as_default(): ## 把新建的图作为默认图
            with tf.gfile.FastGFile(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()    ## 新建GraphDef类对象，用于存储图
                graph_def.ParseFromString(f.read())   ## 解析pb文件的GraphDef对象
                
            train_datareader = read_utils.ReadData(
                    self.LENTH_MAX, mode="train", load_size=self.LOAD_SIZE, vocabulary_size=self.VOCABULARY_SIZE)
            textnum = train_datareader.Test(text)  ### 对输入文本进行规则化处理，编码
            feed_dict = {'text_input':textnum, 'keep_prob':1.0}
            result=tf.import_graph_def(graph_def, input_map=feed_dict, return_elements=['result:0'])      ##无返回类型的操作，把文件中的graph导入默认graph
            # import_graph_def(
            #         graph_def,     --------pb文件解析出来的GraphDef对象
            #         input_map=None,   -----需要映射到图中op的值，这里是用来填放feed_dict的
            #         return_elements=None,--网络需要返回的值
            #         name=None,           --这个操作的名称
            #         op_dict=None,        --已经弃用的选项
            #         producer_op_list=None    --
            #     )
        with tf.Session(graph = graph) as sess:
            last = sess.run(result)
            print(last[0][0])
            return(last[0][0])


if __name__=="__main__":
    # with tf.variable_scope('all') as scope:
    #     test = CNN()
    #     print(type(test))
        # test.CNNTrain(mode = "train")
        # test.test("包装严实物流也很快，下次还会再来")
        # test.test("非常感谢京东商城给予的优质的服务，从仓储管理、物流配送等各方面都是做的非常好的。送货及时，配送员也非常的热情，有时候不方便收件的时候，也安排时间另行配送。同时京东商城在售后")
        # test.CNNTrain(mode = "test", text="包装严实物流也很快，下次还会再来")
        # scope.reuse_variables()
        # test.CNNTrain(mode = "test", text="非常感谢京东商城给予的优质的服务，从仓储管理、物流配送等各方面都是做的非常好的。送货及时，配送员也非常的热情，有时候不方便收件的时候，也安排时间另行配送。同时京东商城在售后")
    test = CNN()
    # test.CNNTrain(mode="train")
    test.test("包装严实物流也很快，下次还会再来")
    test.test("不好，宝宝都不喜欢喝，二阶段的宝宝就喝，这个浪费了。")

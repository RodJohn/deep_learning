# 一、项目介绍 
深度学习在ctr预估中的应用，包括dnn、deepfm、wide&amp;deep模型。

该工程全程仅使用tensorflow低阶API，好处主要有：1、高度可定制化的模型；2、帮助开发者更快更深理解深度神经网络的本质。但相比使用Keras或estimator等高阶API，劣势也很明显：1、代码量较大，对初学者不友好；2、需要开发者对深度神经网络内部构造足够熟悉；3、复杂模型开发难度高。

本人建议，作为深度学习的开发者，要掌握用低阶API写出一些常见的简单深度学习模型，对其内部细节熟悉后，再去使用高级API去开发更复杂先进的模型。

这里附上本人使用estimator的两个项目链接：
    
    https://github.com/R-Stalker/deep_learning_estimator    
    https://github.com/R-Stalker/deep_learning_estimator_labels
分别是单目标和多目标任务，使用高阶API后代码更精简，结构更清晰，包括din、dien、autoint、esmm、mmoe等复杂模型。


# 二、代码目录
##1.主函数

    local_run.py
    local_run_test.py——用于本地测试
    local_run_textline.py——针对textline的数据加载方式（只是为了记录下该方法，可忽略）


##2.utils——工具函数文件夹

    utils/my_utils.py——一些共用函数
    utils/data_loader_load.py——加载明文数据到内存中（只是为了记录下该方法，可忽略。这种方式易实现，适合小数据量，数据量大会爆内存，不推荐。）
    utils/data_loader_textline.py——以textline方式加载数据（只是为了记录下该方法，可忽略。这种方法逐行加载训练数据，数据量再大也只占用极少的内存，但缺点就是速度太慢，不太推荐）
    utils/data_loader.py——加载tfrecord数据（训练数据之间转成tfrecord，然后按照batch_size加载tfrecord数据，兼顾速度与内存资源，强烈推荐）


##3.models——即深度学习模型文件，包括模型构建、训练、预测等

    models/dnn_pipeline——原始dnn模型，使用连续特征和单值离散特，训练数据为tfrecord格式，对应utils/data_loader.py
  
    models/dnn.py——原始dnn模型，使用连续特征和单值离散特征
    models/dnn_cate.py——仅用单值离散特征的dnn模型
    models/dnn_multi.py——使用连续、单值离散、多值离散特征的dnn模型
    models/dnn_multi_cate.py——使用单值离散、多值离散特征的dnn模型
    models/dnn_multi_textline.py——使用连续、单值离散、多值离散特征的dnn模型，这一点和dnn_multi一样。但是这里使用了不同的训练数据加载方式——testline，在此之前4个模型的数据加载方式是直接load在内存中，这种方法的弊端在介绍utils里说过了。
      
    models/deepfm.py——....类似dnn，此处省略
    models/deepfm_cate.py——...
    ...
      
##4.test_model——部分待二次开发模型，没有用于实际工程

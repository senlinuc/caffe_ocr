# 简介
caffe_ocr是一个对现有主流ocr算法研究实验性的项目，目前实现了CNN+BLSTM+CTC的识别架构，并在数据准备、网络设计、调参等方面进行了诸多的实验。代码包含了对lstm、warp-ctc、multi-label等的适配和修改，还有基于inception、restnet、densenet的网络结构。代码是针对windows平台的，linux平台下只需要合并相关的修改合并到caffe代码中即可。
## caffe代码修改
  1. data layer增加了对multi-label的支持<br>
  2. lstm使用的是junhyukoh实现的lstm版本（lstm_layer_Junhyuk.cpp/cu），原版不支持变长输入的识别。输入的shape由(TxN)xH改为TxNxH以适应ctc的输入结构。<br>
  3. WarpCTCLossLayer去掉了对sequence indicators依赖（训练时CNN输出的结构是固定的），简化了网络结构（不需要sequence indicator layer）。<br>
  4. densenet修改了对Reshape没有正确响应的bug，实现了对变长输入预测的支持。<br>
  5. 增加transpose_layer、reverse_layer，实现对CNN feature map与lstm输入shape的适配<br>
## 编译
   1. 安装opencv,boost,cuda,其它依赖库在3rdparty下<br>
   2. caffe-vsproj下为vs2015的工程，配置好依赖库的路径即可编译，编译后会在tools_bin目录下生成训练程序caffe.exe<br>
   3. 相关的依赖dll可从百度网盘下载（http://pan.baidu.com/s/1boOiscJ）<br>
## 实验
  1. 数据准备<br>
  （1）[VGG Synthetic Word Dataset](http://www.robots.ox.ac.uk/~vgg/data/text/)<br>
  （2）自己生成的中文数据<br>
  2. 网络设计<br>
     网络结构在examples/ocr目录下<br>
  3. 主要实验结果<br>
  
各网络在中文数据集上的预测速度、准确率、模型大小对比

| 网格结构 | CPU | GPU | 准确率 | 模型大小 |
| ---------- | -----------| ---------- | -----------| -----------|
| inception-bn-res-blstm | 65.4ms | 11.2ms | 0.92 | 26.9MB |
| resnet-res-blstm	| 64ms	| 10.75ms	| 0.91	| 23.2MB| 
| densenet-res-blstm	| N/A	| 7.73ms	| 0.965	| 22.9MB| 
| densenet-no-blstm	| N/A	| 2.4ms	| 0.97	| 5.6MB| 

说明：（1）CPU是Xeon E3 1230, GPU是1080TI<br>
     （2）densenet使用的是memory-efficient版本，其CPU代码并没有使用blas库，只是实现了原始的卷积操作，速度非常慢，待优化后再做对比。<br>
     （3）“res-blstm”表示残差形式的blstm，“no-blstm”表示没有lstm层，CNN直接对接CTC<br>
     （4）准确率是指整串正确的比例<br>
   4. 一些tricks<br>
    （1） 残差形式的blstm可显著提升准确率，中文数据集上0.94-->0.965<br>
    （2） 网络的CNN部分相对于BLSTM部分可以设置更高的学习率，这样可以显著增加收敛速度<br>
   5. 疑问<br>
    （1）去掉blstm，直接用CNN+CTC的结构，在中文数据集上可以取得更高的准确率（densenet-no-blstm），为什么？<br>
        可能的原因：a）CNN最后一层得到的特征对于字符级别的建模已经具有很好表征，lstm虽然理论上可以通过学习长期依赖关系来学到语言模型级别的信息，但由于某些原因网络并没有学习到这些信息。b)lstm收敛较慢，需要更长的时间才能达到相同的精度。<br>
   6. 现存的问题<br>
    （1）宽度较小的数字、英文字符会出现丢字的情况，如“11”、“ll”，应该是因为CNN特征感受野过大没有看到文字间隙的缘故。<br>
## 提高准确率TODO
  1. 数据方面:增大数据量，文字均匀采样<br>
  2. 网络方面：Attention,STN,辅助loss<br>
## 引用
  1. multi-label的支持(http://blog.csdn.net/hubin232/article/details/50960201)<br>
  2. junhyukoh实现的lstm版本（https://github.com/junhyukoh/caffe-lstm）<br>
  3. caffe-warp-ctc(https://github.com/BVLC/caffe/pull/4681)<br>
  4. memory-efficient densenet(https://github.com/Tongcheng/caffe/)<br>

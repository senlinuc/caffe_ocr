# 简介
caffe_ocr是一个对现有主流ocr算法研究实验性的项目，目前实现了CNN+BLSTM+CTC的识别架构，并在数据准备、网络设计、调参等方面进行了诸多的实验。代码包含了对lstm、warp-ctc、multi-label等的适配和修改，还有基于inception、restnet、densenet的网络结构。代码是针对windows平台的，linux平台下只需要合并相关的修改到caffe代码中即可。
## caffe代码修改
  1. data layer增加了对multi-label的支持<br>
  2. lstm使用的是junhyukoh实现的lstm版本（lstm_layer_Junhyuk.cpp/cu），原版不支持变长输入的识别。输入的shape由(TxN)xH改为TxNxH以适应ctc的输入结构。<br>
  3. WarpCTCLossLayer去掉了对sequence indicators依赖（训练时CNN输出的结构是固定的），简化了网络结构（不需要sequence indicator layer）。<br>
  4. densenet修改了对Reshape没有正确响应的bug，实现了对变长输入预测的支持。<br>
  5. 增加transpose_layer、reverse_layer，实现对CNN feature map与lstm输入shape的适配<br>
## 编译
   1. 安装opencv,boost,cuda,其它依赖库在3rdparty下（包含debug版的lib:http://pan.baidu.com/s/1nvIFojJ）<br>
   2. caffe-vsproj下为vs2015的工程，配置好依赖库的路径即可编译，编译后会在tools_bin目录下生成训练程序caffe.exe<br>
   3. 相关的依赖dll可从百度网盘下载（http://pan.baidu.com/s/1boOiscJ）<br>
## 测试
   1. 编译caffe-vsproj/libClassification,修改自examples\cpp_classification，实现了模型加载、预测等基本功能。<br>
   2. 编译caffe-vsproj/ocr_test，实现了中文、英文ocr模型的测试功能，英文ocr包含了直接预测和词典辅助预测（英文词典文件examples/ocr/lexicon.txt）<br>
## 实验
1. 数据准备<br>
  （1）[VGG Synthetic Word Dataset](http://www.robots.ox.ac.uk/~vgg/data/text/)<br>
  （2）合成的中文数据（https://pan.baidu.com/s/1dFda6R3 ）<br>
  ![](https://github.com/senlinuc/caffe_ocr/blob/master/examples/ocr/20436312_1683447152.jpg)  ![](https://github.com/senlinuc/caffe_ocr/blob/master/examples/ocr/20436328_800384098.jpg)
  ![](https://github.com/senlinuc/caffe_ocr/blob/master/examples/ocr/20436765_2556130357.jpg)  ![](https://github.com/senlinuc/caffe_ocr/blob/master/examples/ocr/20437703_937698549.jpg)
  ![](https://github.com/senlinuc/caffe_ocr/blob/master/examples/ocr/20438234_4066860539.jpg)  ![](https://github.com/senlinuc/caffe_ocr/blob/master/examples/ocr/20440484_2127180630.jpg)
>>*   数据是利用中文语料库（新闻+文言文），通过字体、大小、灰度、模糊、透视、拉伸等变化随机生成的。<br>
>>*   字典中包含汉字、标点、英文、数字共5990个字符（语料字频统计，全角半角合并）<br>
>>*   每个样本固定10个字符，字符随机截取自语料库中的句子<br>
>>*   图片分辨率统一为280x32<br>
>>*   共生成约360万张图片，按9:1分成训练集、验证集，测试集约6万张<br>

2. 网络设计<br>
   网络结构在examples/ocr目录下<br>
3. 主要实验结果<br>

* 英文数据集 VGG Synthetic Word Dataset:

| 网格结构 | predict-CPU | predict-GPU | 准确率-no lexicon | 准确率-lexicon-minctcloss | 模型大小 |
| ---------- | -----------| ---------- | ---------- | -----------| -----------|
| crnn | 67.13ms | 10.28ms | 0.8435 | 0.9163 | 32MB |
| inception-bn-res-blstm | 41.62ms | 8.68ms | 0.7353 | 0.8609 | 15MB |
| densenet-res-blstm	| N/A	| 6.07ms	| 0.7548	| 0.893 | 11MB| 

* 中文数据集:

| 网格结构 | predict-CPU | predict-GPU | 准确率 | 模型大小 |
| ---------- | -----------| ---------- | -----------| -----------|
| inception-bn-res-blstm | 65.4ms | 11.2ms | 0.92 | 26.9MB |
| resnet-res-blstm	| 64ms	| 10.75ms	| 0.91	| 23.2MB| 
| densenet-res-blstm	| N/A	| 7.73ms	| 0.965	| 22.9MB| 
| densenet-no-blstm	| N/A	| 2.4ms	| 0.97	| 5.6MB| 
|densenet-sum-blstm-full-res-blstm|N/A|7.83ms|0.9805|15.5MB|
|densenet-no-blstm-vertical-feature|N/A|3.05ms|0.9816|9.6MB|

中文数据集上训练好的模型：http://pan.baidu.com/s/1i5d5zdN
>说明：<br>
>>*   CPU是Xeon E3 1230, GPU是1080TI<br>
>>*   densenet使用的是memory-efficient版本，其CPU代码并没有使用blas库，只是实现了原始的卷积操作，速度非常慢，待优化后再做对比。<br>
>>*   “res-blstm”表示残差形式的blstm，“no-blstm”表示没有lstm层，CNN直接对接CTC<br>
>>*   准确率是指整串正确的比例,在验证集上统计,"准确率-no lexicon"表示没用词典的准确率，"准确率-lexicon-minctcloss"指先在词典中查找Edit Distance <=2的单词，再选择ctcloss最小的单词作为识别结果<br>
>>*   predict-CPU/GPU为单张图片的预测时间，predict-CPU的后端是openblas，MKL比openblas快约一倍。中文数据集上图片分辨率为280x32，英文数据集100x32
>>*   densenet-sum-blstm-full-res-blstm相对于densenet-res-blstm有两点改动：（1）两个lstm结合成blstm的方式由concat改为sum；（2）两层blstm都采用残差方式连接（CNN最后的Channel数改为与blstm结点数相同），最后得到了最高的准确率。
>>*   densenet-no-blstm-vertical-feature相对于densenet-no-blstm去掉了1x4的pooling层，使得CNN最后一层的feature maps的高度为4，更好的保留笔画在垂直方向的变化信息，再reshape BxCxHxW --> Bx(CxH)x1xW串成高度为1的feature maps, 最后准确率上略好于之前的最好结果，可见CNN feature方面还有很多可挖掘的地方。
4. 一些tricks<br>
  （1） 残差形式的blstm可显著提升准确率，中文数据集上0.94-->0.965，两层BLSTM都用残差后又提升到了0.9805<br>
  （2） 网络的CNN部分相对于BLSTM部分可以设置更高的学习率，这样可以显著增加收敛速度<br>
5. 疑问<br>
  （1）去掉blstm，直接用CNN+CTC的结构，在中文数据集上也可以取得很高的准确率（densenet-no-blstm），为什么？<br>
      可能的原因：a）CNN最后一层得到的特征对于字符级别的建模已经具有很好表征，b)lstm收敛较慢，需要更长的时间才能达到相同的精度。<br>
6. 现存的问题<br>
  （1）宽度较小的数字、英文字符会出现丢字的情况，如“11”、“ll”，应该是因为CNN特征感受野过大没有看到文字间隙的缘故。<br>
## 提高准确率TODO
  1. 数据方面: 增大数据量，语料库均匀采样(https://yq.aliyun.com/articles/109555?t=t1)<br>
  2. 网络方面：增大网络（train、test loss很接近，现有网络没有过拟合），Attention,STN,辅助loss<br>
## 引用
  1. crnn(https://arxiv.org/abs/1507.05717)<br> 
  2. multi-label的支持(http://blog.csdn.net/hubin232/article/details/50960201)<br>
  3. junhyukoh实现的lstm版本（https://github.com/junhyukoh/caffe-lstm）<br>
  4. caffe-warp-ctc(https://github.com/BVLC/caffe/pull/4681)<br>
  5. memory-efficient densenet(https://github.com/Tongcheng/caffe/)<br>
  6. 语料库（http://www.sogou.com/labs/resource/cs.php）<br>
 

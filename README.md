# 简介
caffe_ocr是一个对现有主流ocr算法研究实验性的项目，目前实现了CNN+BLSTM+CTC的识别架构，并在数据准备、网络设计、调参等方面进行了诸多的实验。代码包含了对lstm、warp-ctc、multi-label等的适配和修改，还有基于inception、restnet、densenet的网络结构。目前代码是针对windows平台，linux平台下只需要并相关的修改合并到caffe代码中即可。
## caffe代码修改
  1. data layer增加了对multi-label的支持<br>
  2. lstm使用的是junhyukoh实现的lstm版本（lstm_layer_Junhyuk.cpp/cu），原版不支持变长输入的识别。输入的shape由(TxN)xH改为TxNxH以适应ctc的输入结构。<br>
  3. WarpCTCLossLayer去掉了对sequence indicators依赖（训练时CNN输出的结构是固定的），简化了网络结构（不需要sequence indicator layer）。<br>
  4. densenet修改了对Reshape没有正确响应的bug，实现了对变长输入预测的支持。<br>
  5. 增加transpose_layer、reverse_layer，实现对CNN feature map与lstm输入shape的适配<br>
## 主要实验结果
## 提高准确率TODO
  1. 数据方面:增大数据量，文字均匀采样<br>
  2. 网络方面：Attention,STN,辅助loss<br>
## 引用
  1. multi-label的支持(http://blog.csdn.net/hubin232/article/details/50960201)<br>
  2. junhyukoh实现的lstm版本（https://github.com/junhyukoh/caffe-lstm）<br>
  3. caffe-warp-ctc( )<br>
  4. memory-efficient densenet(https://github.com/Tongcheng/caffe/)<br>

// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;


namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	top_k_ = this->layer_param_.accuracy_param().top_k();

	has_ignore_label_ =
		this->layer_param_.accuracy_param().has_ignore_label();
	if (has_ignore_label_) {
		ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
	}
	/*
  CHECK_EQ(bottom.size(), 2)
    << "MultiLabelAccuracy Layer takes two blobs as input.";
  CHECK_LE(top->size(), 1)
    << "MultiLabelAccuracy Layer takes 0/1 output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number of instances";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
    << "The data and label should have the same number of channels";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "The data and label should have the same height";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "The data and label should have the same width";
  if (top->size() == 1) {
    // If top is used then it will contain:
    // top[0] = Sensitivity (TP/P),
    // top[1] = Specificity (TN/N),
    // top[2] = Harmonic Mean of Sens and Spec, 2/(P/TP+N/TN),
    // top[3] = Loss
    (*top)[0]->Reshape(1, 4, 1, 1);
  }
  */
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
		<< "top_k must be less than or equal to the number of classes.";
	label_axis_ =
		bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
	outer_num_ = bottom[0]->count(0, label_axis_);
	inner_num_ = bottom[0]->count(label_axis_ + 1);

	//label个数，每个样本多个label，不是label的可能性数
	label_num_ = label_axis_ < bottom[1]->num_axes() ?
		bottom[1]->shape(label_axis_) : 1;
	CHECK_EQ(outer_num_, bottom[1]->count(0, label_axis_));
// 	if (label_axis_ < bottom[1]->num_axes()) {
// 		CHECK_EQ(inner_num_, bottom[1]->count(label_axis_ + 1));
// 	}
// 	else {
// 		CHECK_EQ(inner_num_, 1);
// 	}
	//  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
	//      << "Number of labels must match number of predictions; "
	//      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
	//      << "label count (number of labels) must be N*H*W, "
	//      << "with integer values in {0, 1, ..., C-1}.";

	vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
	top[0]->Reshape(top_shape);
	if (top.size() > 1) {
		// Per-class accuracy is a vector; 1 axes.
		vector<int> top_shape_per_class(1);
		top_shape_per_class[0] = bottom[0]->shape(label_axis_);
		top[1]->Reshape(top_shape_per_class);
		nums_buffer_.Reshape(top_shape_per_class);
	}
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
	Dtype accuracy = 0;
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	const int dim = bottom[0]->count() / outer_num_;
	const int ch = bottom[0]->channels();
	if (top.size() > 1) {
		caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
		caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
	}
	int count = 0;
	for (int i = 0; i < outer_num_; ++i) {
		for (int l = 0; l < label_num_; l++) {
			const int label_value =
				static_cast<int>(bottom_label[i * inner_num_ + l]);
			if (has_ignore_label_ && label_value == ignore_label_) {
				continue;
			}
			if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
			DCHECK_GE(label_value, 0);
			DCHECK_LT(label_value, ch);
			// Top-k accuracy
			std::vector<std::pair<Dtype, int> > bottom_data_vector;
			for (int k = 0; k < ch; ++k) {
				bottom_data_vector.push_back(std::make_pair(
					bottom_data[i * dim + k * inner_num_ + l], k));
			}
#ifdef _DEBUG
			float sum = 0;
			for (size_t m=0;m<bottom_data_vector.size();m++)
			{
				sum += bottom_data_vector[m].first;
			}
#endif
			std::partial_sort(
				bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
				bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
			// check if true label is in top k predictions
			for (int k = 0; k < top_k_; k++) {
				if (bottom_data_vector[k].second == label_value) {
					++accuracy;
					if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
					break;
				}
			}
			++count;
		}
	}

	// LOG(INFO) << "Accuracy: " << accuracy;
	top[0]->mutable_cpu_data()[0] = accuracy / count;
	if (top.size() > 1) {
		for (int i = 0; i < top[1]->count(); ++i) {
			top[1]->mutable_cpu_data()[i] =
				nums_buffer_.cpu_data()[i] == 0 ? 0
				: top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
		}
	}
  // MultiLabelAccuracy should not be used as a loss function.
}



INSTANTIATE_CLASS(MultiLabelAccuracyLayer);
REGISTER_LAYER_CLASS(MultiLabelAccuracy);

}  // namespace caffe

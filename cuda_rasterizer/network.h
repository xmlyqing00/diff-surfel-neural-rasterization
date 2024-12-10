#ifndef NETWORK
#define NETWORK

#include <torch/extension.h>
#include <glm/glm.hpp>


typedef long long ll;
const int input_dim = 2;
const int hidden_dim = 4;
const int output_dim = 4;
const ll l1_lw_len = input_dim * hidden_dim + hidden_dim;
const ll l1_mg_len = input_dim * hidden_dim + hidden_dim;
const ll l1_lw2_len = hidden_dim * hidden_dim + hidden_dim;
const ll lout_lw_len = hidden_dim * output_dim + output_dim;

class Network {

public:
	const float* l1_lw;
	const float* l1_mg;
	const float* l1_lw2;
	const float* lout_lw;

	int gauss_num;
	int l1_lw_size, l1_mg_size, l1_lw2_size, lout_lw_size;
    
    Network(const torch::Tensor& l1_lw_, const torch::Tensor& l1_mg_, const torch::Tensor& l1_lw2_, const torch::Tensor& lout_lw_) {

		gauss_num = l1_lw_.size(0);
		
		// contiguous() is used to ensure that the tensor is stored in a contiguous chunk of memory
		l1_lw = l1_lw_.contiguous().data<float>(); // (N, hidden_dim, in_dim + 1)
		l1_mg = l1_mg_.contiguous().data<float>();  // (N, hidden_dim, in_dim + 1)
		l1_lw2 = l1_lw2_.contiguous().data<float>();  // (N, hidden_dim, hidden_dim + 1)
		lout_lw = lout_lw_.contiguous().data<float>();  // (N, out_dim, hidden_dim + 1)
	}

	__device__ void get_params(
		int idx, 
		float * l1_weight, float * l1_bias,
		float * lout_weight, float * lout_bias
	) const {
		int weight_size = hidden_dim * input_dim;
		int offset = idx * (weight_size + hidden_dim);
		for (int i = 0; i < weight_size; i++) l1_weight[i] = l1_lw[offset + i];
		offset += weight_size;
		for (int i = 0; i < hidden_dim; i++) l1_bias[i] = l1_lw[offset + i];
		offset += hidden_dim;

		offset += idx * (hidden_dim * (input_dim + 1));
		offset += idx * (hidden_dim * (hidden_dim + 1));

		weight_size = hidden_dim * output_dim;
		offset += idx * (weight_size + output_dim);
		for (int i = 0; i < weight_size; i++) lout_weight[i] = lout_lw[offset + i];
		offset += weight_size;
		for (int i = 0; i < output_dim; i++) lout_bias[i] = lout_lw[offset + i];
	}

	__device__ void linear(
		const float * input, float * output,
		float * weight, float * bias,
		const int in_dim, const int out_dim
	) const {
		for (int i = 0; i < out_dim; i++) {
			output[i] = bias[i];
		}

		for (int i = 0; i < in_dim; i++) {
			const int offset = i * out_dim;
			for (int j = 0; j < out_dim; j++) {
				output[j] += weight[offset + j] * input[i];
			}
		}
	}

	__device__ void relu(
		float * input, const int dim
	) const {
		for (int i = 0; i < dim; i++) {
			input[i] = fmaxf(0.0f, input[i]);
		}
	}
	
};


class NetworkGrad {

public:
	float * dL_l1_lw;
	float * dL_l1_mg;
	float * dL_l1_lw2;
	float * dL_lout_lw;

	NetworkGrad (
		torch::Tensor &dL_l1_lw_, torch::Tensor &dL_l1_mg_, torch::Tensor &dL_l1_lw2_, torch::Tensor &dL_lout_lw_
	) {
		dL_l1_lw = dL_l1_lw_.contiguous().data_ptr<float>();
		dL_l1_mg = dL_l1_mg_.contiguous().data_ptr<float>();
		dL_l1_lw2 = dL_l1_lw2_.contiguous().data_ptr<float>();
		dL_lout_lw = dL_lout_lw_.contiguous().data_ptr<float>();

		// printf("dL_l1_lw.shape: %d, %d, %d\n", dL_l1_lw.size(0), dL_l1_lw.size(1), dL_l1_lw.size(2));
		// printf("dL_l1_mg.shape: %d, %d, %d\n", dL_l1_mg.size(0), dL_l1_mg.size(1), dL_l1_mg.size(2));
		// printf("dL_l1_lw2.shape: %d, %d, %d\n", dL_l1_lw2.size(0), dL_l1_lw2.size(1), dL_l1_lw2.size(2));
		// printf("dL_lout_lw.shape: %d, %d, %d\n", dL_lout_lw.size(0), dL_lout_lw.size(1), dL_lout_lw.size(2));

		// Check memory
		// for (int i = 0; i < dL_l1_lw.size(0) * dL_l1_lw.size(1) * dL_l1_lw.size(2); i++) {
		// 	printf("dL_l1_lw[%d]: %f\n", i, this->dL_l1_lw[i]);
		// }
		// for (int i = 0; i < dL_l1_mg.size(0) * dL_l1_mg.size(1) * dL_l1_mg.size(2); i++) {
		// 	printf("dL_l1_mg[%d]: %f\n", i, this->dL_l1_mg[i]);
		// }
		// for (int i = 0; i < dL_l1_lw2.size(0) * dL_l1_lw2.size(1) * dL_l1_lw2.size(2); i++) {
		// 	printf("dL_l1_lw2[%d]: %f\n", i, this->dL_l1_lw2[i]);
		// }
		// for (int i = 0; i < dL_lout_lw.size(0) * dL_lout_lw.size(1) * dL_lout_lw.size(2); i++) {
		// 	printf("dL_lout_lw[%d]: %f\n", i, this->dL_lout_lw[i]);
		// }
	}

	__device__ void grad(
		const float * input, int idx,
		const float * dL_dcolor, const Network * net
	) {
		int weight_size = hidden_dim * input_dim;
		int offset = idx * (weight_size + hidden_dim);
		for (int i = 0; i < input_dim; i++) {
			for (int j = 0; j < hidden_dim; j++) {
				// printf("idx: %d, i: %d, j: %d, offset: %d, val: %d\n", idx, i, j, offset + i * input_dim + j, offset + i * input_dim + j);
				// printf("val: %.3f\n", dL_l1_lw[offset + i * input_dim + j]);
				atomicAdd(&dL_l1_lw[offset + i * hidden_dim + j], dL_dcolor[j] * input[i]);
			}
		}
		offset += weight_size;
		// printf("offset: %d\n", offset);
		for (int i = 0; i < hidden_dim; i++) {
			atomicAdd(&dL_l1_lw[offset + i], dL_dcolor[i]);
		}
	}
};


#endif

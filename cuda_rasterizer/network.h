#ifndef NETWORK
#define NETWORK

#include <torch/extension.h>
#include <glm/glm.hpp>
#include "auxiliary.h"

class Network {

public:
	const float* l1_lw;
	const float* l1_mg;
	const float* l1_lw2;
	const float* lout_lw;

	int gauss_num;
	int l1_lw_size, l1_mg_size, l1_lw2_size, lout_lw_size;
    
    Network(const torch::Tensor& l1_lw, const torch::Tensor& l1_mg, const torch::Tensor& l1_lw2, const torch::Tensor& lout_lw) {

		gauss_num = l1_lw.size(0);
		
		// contiguous() is used to ensure that the tensor is stored in a contiguous chunk of memory
		this->l1_lw = l1_lw.contiguous().data<float>(); // (N, hidden_dim, in_dim + 1)
		this->l1_mg = l1_mg.contiguous().data<float>();  // (N, hidden_dim, in_dim + 1)
		this->l1_lw2 = l1_lw2.contiguous().data<float>();  // (N, hidden_dim, hidden_dim + 1)
		this->lout_lw = lout_lw.contiguous().data<float>();  // (N, out_dim, hidden_dim + 1)
	}

	// __device__ void get_params(
	// 	int idx, 
	// 	float * l1_weight, float * l1_bias
	// ) const {
	// 	int weight_size = hidden_dim * input_dim;
	// 	int offset = idx * (weight_size + hidden_dim);
	// 	for (int i = 0; i < weight_size; i++) l1_weight[i] = l1_lw[offset + i];
	// 	offset += weight_size;
	// 	for (int i = 0; i < hidden_dim; i++) l1_bias[i] = l1_lw[offset + i];
	// }

	// __device__ void linear(
	// 	const float * input, float * output,
	// 	float * weight, float * bias,
	// 	const int in_dim, const int out_dim
	// ) const {
	// 	for (int i = 0; i < out_dim; i++) {
	// 		output[i] = bias[i];
	// 		int offset = i * in_dim;
	// 		for (int j = 0; j < in_dim; j++) {
	// 			output[i] += input[j] * weight[offset + j];
	// 		}
	// 	}
	// }

	
};
#endif

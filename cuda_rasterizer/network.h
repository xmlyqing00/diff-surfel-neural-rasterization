#ifndef NETWORK
#define NETWORK

#include <torch/extension.h>
#include <glm/glm.hpp>


typedef long long ll;
const int input_dim = 2;
const int hidden_dim = 4;
const int output_dim = 4;


class Network {

public:
	float * l1_weight;
	float * l1_bias;
	float * l1_mu;
	float * l1_gamma;
	float * l1_weight2;
	float * l1_bias2;
	float * lout_weight;
	float * lout_bias;

	float * dL_l1_weight;
	float * dL_l1_bias;
	float * dL_l1_mu;
	float * dL_l1_gamma;
	float * dL_l1_weight2;
	float * dL_l1_bias2;
	float * dL_lout_weight;
	float * dL_lout_bias;
    
    __device__ Network() {}

	__device__ void forward(const float * input, float * output) const {

		float l1_out[hidden_dim];
		linear(input, l1_out, l1_weight, l1_bias, input_dim, hidden_dim);
		relu(l1_out, l1_out, hidden_dim);

		float l1_out2[hidden_dim];
		linear(l1_out, l1_out2, l1_weight2, l1_bias2, hidden_dim, hidden_dim);
		relu(l1_out2, l1_out2, hidden_dim);

		linear(l1_out2, output, lout_weight, lout_bias, hidden_dim, output_dim);
	}

	__device__ void backward(const float * input, const float * dL_out, float * dL_input) {

		// forward
		float l1_out[hidden_dim];
		linear(input, l1_out, l1_weight, l1_bias, input_dim, hidden_dim);
		float l1_out_act[hidden_dim];
		relu(l1_out, l1_out_act, hidden_dim);

		float l1_out2[hidden_dim];
		linear(l1_out_act, l1_out2, l1_weight2, l1_bias2, hidden_dim, hidden_dim);
		float l1_out2_act[hidden_dim];
		relu(l1_out2, l1_out2_act, hidden_dim);

		// gradient for output layer
		for (int i = 0; i < output_dim; i++) {
			atomicAdd(dL_lout_bias+i, dL_out[i]);
		}
		for (int i = 0; i < hidden_dim; i++) {
			for (int j = 0; j < output_dim; j++) {
				atomicAdd(dL_lout_weight+ i * output_dim + j, dL_out[j] * l1_out2_act[i]);
			}
		}
		float dL_l1_out2_act[hidden_dim];
		for (int i = 0; i < hidden_dim; i++) {
			dL_l1_out2_act[i] = 0;
			for (int j = 0; j < output_dim; j++) {
				dL_l1_out2_act[i] += dL_out[j] * lout_weight[i * output_dim + j];
			}
		}

		// gradient for relu
		float dL_l1_out2[hidden_dim];
		for (int i = 0; i < hidden_dim; i++) {
			dL_l1_out2[i] = l1_out2[i] > 0 ? dL_l1_out2_act[i] : 0;
		}

		// gradient for l1 layer 2
		for (int i = 0; i < hidden_dim; i++) {
			atomicAdd(dL_l1_bias2+i, dL_l1_out2[i]);
		}
		for (int i = 0; i < hidden_dim; i++) {
			for (int j = 0; j < hidden_dim; j++) {
				atomicAdd(dL_l1_weight2+ i * hidden_dim + j, dL_l1_out2[j] * l1_out_act[i]);
			}
		}
		float dL_l1_out_act[hidden_dim];
		for (int i = 0; i < hidden_dim; i++) {
			dL_l1_out_act[i] = 0;
			for (int j = 0; j < hidden_dim; j++) {
				dL_l1_out_act[i] += dL_l1_out2[j] * l1_weight2[i * hidden_dim + j];
			}
		}

		// gradient for relu
		float dL_l1_out[hidden_dim];
		for (int i = 0; i < hidden_dim; i++) {
			dL_l1_out[i] = l1_out[i] > 0 ? dL_l1_out_act[i] : 0;
		}

		// gradient for l1 layer
		for (int i = 0; i < hidden_dim; i++) {
			atomicAdd(dL_l1_bias+i, dL_l1_out[i]);
		}
		for (int i = 0; i < input_dim; i++) {
			for (int j = 0; j < hidden_dim; j++) {
				atomicAdd(dL_l1_weight+ i * hidden_dim + j, dL_l1_out[j] * input[i]);
			}
		}
		for (int i = 0; i < input_dim; i++) {
			dL_input[i] = 0;
			for (int j = 0; j < hidden_dim; j++) {
				dL_input[i] += dL_l1_out[j] * l1_weight[i * hidden_dim + j];
			}
		}

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
		const float * input, float * output, const int dim
	) const {
		for (int i = 0; i < dim; i++) {
			output[i] = fmaxf(0.0f, input[i]);
		}
	}
	
};

class Params {

private:
	int l1_lw_size = (input_dim + 1) * hidden_dim;
	int l1_mg_size = (input_dim + 1) * hidden_dim;
	int l1_lw2_size = (hidden_dim + 1) * hidden_dim;
	int lout_lw_size = (hidden_dim + 1) * output_dim;
	int size_per_gauss = l1_lw_size + l1_mg_size + l1_lw2_size + lout_lw_size;

	int l1_bias_start = input_dim * hidden_dim;
	int l1_gamma_start = input_dim * hidden_dim;
	int l1_bias2_start = hidden_dim * hidden_dim;
	int lout_linear_bias_start = hidden_dim * output_dim;

public:
	float* l1_lw = nullptr;
	float* l1_mg = nullptr;
	float* l1_lw2 = nullptr;
	float* lout_lw = nullptr;

	float * dL_l1_lw = nullptr;
	float * dL_l1_mg = nullptr;
	float * dL_l1_lw2 = nullptr;
	float * dL_lout_lw = nullptr;

	Params() {}
	
    void set_params(const torch::Tensor& l1_lw_, const torch::Tensor& l1_mg_, const torch::Tensor& l1_lw2_, const torch::Tensor& lout_lw_) {

		// contiguous() is used to ensure that the tensor is stored in a contiguous chunk of memory
		l1_lw = l1_lw_.contiguous().data<float>(); // (N, hidden_dim, in_dim + 1)
		l1_mg = l1_mg_.contiguous().data<float>();  // (N, hidden_dim, in_dim + 1)
		l1_lw2 = l1_lw2_.contiguous().data<float>();  // (N, hidden_dim, hidden_dim + 1)
		lout_lw = lout_lw_.contiguous().data<float>();  // (N, out_dim, hidden_dim + 1)
	}

	void set_grads(torch::Tensor &dL_l1_lw_, torch::Tensor &dL_l1_mg_, torch::Tensor &dL_l1_lw2_, torch::Tensor &dL_lout_lw_) {
		dL_l1_lw = dL_l1_lw_.contiguous().data_ptr<float>();
		dL_l1_mg = dL_l1_mg_.contiguous().data_ptr<float>();
		dL_l1_lw2 = dL_l1_lw2_.contiguous().data_ptr<float>();
		dL_lout_lw = dL_lout_lw_.contiguous().data_ptr<float>();
	}

	__device__ void get_params(int idx, Network &net, bool get_grad) const {

		// l1_weight and l1_bias
		int offset = idx * l1_lw_size;
		net.l1_weight = l1_lw + offset;
		net.l1_bias = l1_lw + offset + l1_bias_start;

		// l1_mu and l1_gamma
		offset = idx * l1_mg_size;
		net.l1_mu = l1_mg + offset;
		net.l1_gamma = l1_mg + offset + l1_gamma_start;

		// l1_weight2 and l1_bias2
		offset = idx * l1_lw2_size;
		net.l1_weight2 = l1_lw2 + offset;
		net.l1_bias2 = l1_lw2 + offset + l1_bias2_start;

		// lout_weight and lout_bias
		offset = idx * lout_lw_size;
		net.lout_weight = lout_lw + offset;
		net.lout_bias = lout_lw + offset + lout_linear_bias_start;

		if (get_grad) {
			offset = idx * l1_lw_size;
			net.dL_l1_weight = dL_l1_lw + offset;
			net.dL_l1_bias = dL_l1_lw + offset + l1_bias_start;

			offset = idx * l1_mg_size;
			net.dL_l1_mu = dL_l1_mg + offset;
			net.dL_l1_gamma = dL_l1_mg + offset + l1_gamma_start;

			offset = idx * l1_lw2_size;
			net.dL_l1_weight2 = dL_l1_lw2 + offset;
			net.dL_l1_bias2 = dL_l1_lw2 + offset + l1_bias2_start;

			offset = idx * lout_lw_size;
			net.dL_lout_weight = dL_lout_lw + offset;
			net.dL_lout_bias = dL_lout_lw + offset + lout_linear_bias_start;
		}
	}
};



#endif

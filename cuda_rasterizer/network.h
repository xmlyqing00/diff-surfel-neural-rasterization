#ifndef CUDA_RASTERIZER_NETWORK_H
#define CUDA_RASTERIZER_NETWORK_H

#include <torch/extension.h>
#include <glm/glm.hpp>
#include <cmath>

typedef long long ll;
#define GABOR_IN_DIM 2
#define GABOR_HIDDEN_DIM 4
#define GABOR_OUT_DIM 3
#define GABOR_LAYER_NUM 1


class LinearLayer {
public:
	float * weight, * bias;
	float * dweight, * dbias;
	int in_dim, out_dim;

	__device__ LinearLayer() {}
	__device__ LinearLayer(int in_dim, int out_dim) : in_dim(in_dim), out_dim(out_dim) {};
	
	__device__ void forward(const float * input, float * output) const {
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

	__device__ void backward(const float * input, const float * dL_dout, float * dL_din) {
		for (int i = 0; i < in_dim; i++) {
			int offset = i * out_dim;
			float dL_din_sum = 0;
			for (int j = 0; j < out_dim; j++) {
				atomicAdd(&dweight[offset + j], dL_dout[j] * input[i]);
				dL_din_sum += dL_dout[j] * weight[offset + j];
			}
			dL_din[i] = dL_din_sum;
		}
		for (int j = 0; j < out_dim; j++) {
			atomicAdd(&dbias[j], dL_dout[j]);
		}
	}
};


class GaborInterVars {
public:
	float x2_sum = 0;
	float D[GABOR_HIDDEN_DIM];
	float linear_out[GABOR_HIDDEN_DIM];
	float sin_term[GABOR_HIDDEN_DIM];
	float exp_term[GABOR_HIDDEN_DIM];

	__device__ GaborInterVars() {}
	__device__ void print(float u, float v) {
		printf("uv: %.3f %.3f, x2_sum: %.8f\n", u, v, x2_sum);
		for (int i = 0; i < GABOR_HIDDEN_DIM; i++) {
			printf("uv: %.3f %.3f, D[%d]: %.8f, linear_out[%d]: %.8f, sin_term[%d]: %.8f, exp_term[%d]: %.8f\n", 
				u, v,
				i, D[i], i, linear_out[i], i, sin_term[i], i, exp_term[i]
			);
		}
	}
};


class GaborLayer {
public:
	float * mu, * gamma;
	float * dmu, * dgamma;
	LinearLayer linear;
	
	__device__ GaborLayer(): linear(GABOR_IN_DIM, GABOR_HIDDEN_DIM) {}

	__device__ void forward(
		const float * input, float * output, GaborInterVars &inter_vars
	) const {
		
		inter_vars.x2_sum = 0;
		for (int i = 0; i < GABOR_IN_DIM; i++) {
			inter_vars.x2_sum += input[i] * input[i];
		}

		for (int j = 0; j < GABOR_HIDDEN_DIM; j++) {
			float mu_sum = 0, x_mu_sum = 0;
			for (int i = 0; i < GABOR_IN_DIM; i++) {
				int offset = i * GABOR_HIDDEN_DIM;
				mu_sum += mu[offset + j] * mu[offset + j];
				x_mu_sum += input[i] * mu[offset + j];
			}
			inter_vars.D[j] = inter_vars.x2_sum + mu_sum - 2 * x_mu_sum;
		}

		linear.forward(input, inter_vars.linear_out);

		for (int j = 0; j < GABOR_HIDDEN_DIM; j++) {
			inter_vars.sin_term[j] = sinf(inter_vars.linear_out[j]);
			inter_vars.exp_term[j] = expf(-0.5f * inter_vars.D[j] * gamma[j]);
			output[j] = inter_vars.sin_term[j] * inter_vars.exp_term[j];
		}
	}

	__device__ void backward(
		const float * input, const float * dL_dout, float * dL_din, const GaborInterVars &inter_vars
	) {

		float dL_dD[GABOR_HIDDEN_DIM] = {0};
		float dL_dlin[GABOR_HIDDEN_DIM] = {0};
		for (int j = 0; j < GABOR_HIDDEN_DIM; j++) {
			dL_dlin[j] = dL_dout[j] * inter_vars.exp_term[j] * cosf(inter_vars.linear_out[j]);
			dL_dD[j] = dL_dout[j] * inter_vars.sin_term[j] * inter_vars.exp_term[j] * (-0.5f * gamma[j]);
			atomicAdd(&dgamma[j], dL_dout[j] * inter_vars.sin_term[j] * inter_vars.exp_term[j] * (-0.5f * inter_vars.D[j]));
		}

		float local_din[GABOR_IN_DIM] = {0};
		linear.backward(input, dL_dlin, local_din);

		for (int j = 0; j < GABOR_HIDDEN_DIM; j++) {
			for (int i = 0; i < GABOR_IN_DIM; i++) {
				int off = i * GABOR_HIDDEN_DIM + j;
				atomicAdd(&dmu[off], dL_dD[j] * 2.0f * (mu[off] - input[i]));
				dL_din[i] += dL_dD[j] * 2.0f * (input[i] - mu[off]);
			}
		}

		for (int i = 0; i < GABOR_IN_DIM; i++) {
			dL_din[i] += local_din[i];
		}
	}

};


class Network {

public:
	GaborLayer gabor_layers[GABOR_LAYER_NUM + 1];
	LinearLayer linear_layers[GABOR_LAYER_NUM];
	LinearLayer out_linear;

    __device__ Network() {
		for (int i = 0; i < GABOR_LAYER_NUM + 1; i++) {
			gabor_layers[i] = GaborLayer();
		}
		for (int i = 0; i < GABOR_LAYER_NUM; i++) {
			linear_layers[i] = LinearLayer(GABOR_HIDDEN_DIM, GABOR_HIDDEN_DIM);
		}
		out_linear = LinearLayer(GABOR_HIDDEN_DIM, GABOR_OUT_DIM);
	}

	__device__ void forward(const float input[GABOR_IN_DIM], float output[GABOR_OUT_DIM], bool debug) const {
		
		float filter_out[GABOR_HIDDEN_DIM];
		GaborInterVars gabor_inter_vars;
		gabor_layers[0].forward(input, filter_out, gabor_inter_vars);
		// if (debug) {
		// 	printf("input: %.3f %.3f, filter_out: %.3f %.3f\n", input[0], input[1], filter_out[0], filter_out[1]);
		// }

		float filter_linear_out[GABOR_HIDDEN_DIM];
		for (int i = 1; i <= GABOR_LAYER_NUM; i++) {
			linear_layers[i - 1].forward(filter_out, filter_linear_out);
	
			float filter1_out[GABOR_HIDDEN_DIM];
			gabor_layers[i].forward(input, filter1_out, gabor_inter_vars);
			if (debug) {
				printf("input: %.3f %.3f, filter1_out: %.3f %.3f\n", input[0], input[1], filter1_out[0], filter1_out[1]);
			}
			for (int j = 0; j < GABOR_HIDDEN_DIM; j++) {
				filter_out[j] = filter1_out[j] * filter_linear_out[j];
			}

		}

		out_linear.forward(filter_out, output);
		// if (debug) {
		// 	printf("input: %.3f %.3f, output: %.3f %.3f %.3f\n", input[0], input[1], output[0], output[1], output[2]);
		// }

		// float tmp[GABOR_HIDDEN_DIM];
		// for (int j = 0; j < GABOR_HIDDEN_DIM; j++) {
		// 	tmp[j] = sinf(input[j]);
		// }
		// out_linear.forward(tmp, output);

	}

	__device__ void forward_and_save_inter_vars(
		const float * input, float * output, 
		float filter_out[][GABOR_HIDDEN_DIM], float filter_linear_out[][GABOR_HIDDEN_DIM], float mix_out[][GABOR_HIDDEN_DIM], 
		GaborInterVars gabor_inter_vars[GABOR_LAYER_NUM + 1]
	) {
		gabor_layers[0].forward(input, filter_out[0], gabor_inter_vars[0]);

		int i = 1;
		// int offset_cur = i * GABOR_HIDDEN_DIM;
		// int offset_last = offset_cur - GABOR_HIDDEN_DIM;
		linear_layers[i - 1].forward(filter_out[i - 1], filter_linear_out[i - 1]);
	
		gabor_layers[i].forward(input, filter_out[i], gabor_inter_vars[i]);
			
		for (int j = 0; j < GABOR_HIDDEN_DIM; j++) {
			mix_out[0][j] = filter_out[1][j] * filter_linear_out[0][j];
		}

		out_linear.forward(mix_out[0], output);
	}

	__device__ void backward(
		const float * input, const float * dL_dcolor, float * dL_input, 
		float filter_out[][GABOR_HIDDEN_DIM], float filter_linear_out[][GABOR_HIDDEN_DIM], float mix_out[][GABOR_HIDDEN_DIM], 
		GaborInterVars gabor_inter_vars[GABOR_LAYER_NUM + 1],
		bool debug
	) {

		// float filter_out[GABOR_LAYER_NUM + 1][GABOR_HIDDEN_DIM];
		// float filter_linear_out[GABOR_LAYER_NUM][GABOR_HIDDEN_DIM];
		// float mix_out[GABOR_LAYER_NUM][GABOR_HIDDEN_DIM];

		// GaborInterVars gabor_inter_vars[GABOR_LAYER_NUM + 1];
		// gabor_layers[0].forward(input, filter_out[0], gabor_inter_vars[0]);
		
		// for (int i = 1; i <= GABOR_LAYER_NUM; i++) {
		// 	if (i == 1) {
		// 		linear_layers[i - 1].forward(filter_out[i - 1], filter_linear_out[i - 1]);
		// 	} else {
		// 		linear_layers[i - 1].forward(mix_out[i - 2], filter_linear_out[i - 1]);
		// 	}
		// 	gabor_layers[i].forward(input, filter_out[i], gabor_inter_vars[i]);
		// 	for (int j = 0; j < GABOR_HIDDEN_DIM; j++) {
		// 		mix_out[i - 1][j] = filter_out[i][j] * filter_linear_out[i - 1][j];
		// 	}
		// }
		// linear_layers[0].forward(filter_out[0], filter_linear_out[0]);
		// if (debug) {
		// 	printf("back in: %.8f %.8f, filter_out: %.8f %.8f, filter_linear_out: %.8f %.8f\n", input[0], input[1], filter_out[0][0], filter_out[0][1], filter_linear_out[0][0], filter_linear_out[0][1]);
		// 	printf("back linear weights %.8f %.8f %.8f %.8f, bias %.8f %.8f\n", linear_layers[0].weight[0], linear_layers[0].weight[1], linear_layers[0].weight[2], linear_layers[0].weight[3], linear_layers[0].bias[0], linear_layers[0].bias[1]);
		// }
		
		// gabor_layers[1].forward(input, filter_out[1], gabor_inter_vars[1]);
		// for (int j = 0; j < GABOR_HIDDEN_DIM; j++) {
			// mix_out[0][j] = filter_out[1][j] * filter_linear_out[0][j];
		// }
		// if (debug) {
		// 	printf("back in: %.8f %.8f, mix_filter_out: %.8f %.8f, filter_out1 %.8f, %.8f, filter_linear_out: %.8f %.8f\n", 
		// 	input[0], input[1], mix_out[0][0], mix_out[0][1], filter_out[1][0], filter_out[1][1], filter_linear_out[0][0], filter_linear_out[0][1]
		// 		);
		// }

		float dL_dmix_out[GABOR_HIDDEN_DIM] = {0};
		out_linear.backward(mix_out[0], dL_dcolor, dL_dmix_out);

		// if (debug) {
		// 	printf("back in %.8f %.8f, dL_dcolor: %.8f %.8f %.8f, dL_dmix_out: %.8f %.8f %.8f\n", input[0], input[1], dL_dcolor[0], dL_dcolor[1], dL_dcolor[2], dL_dmix_out[0], dL_dmix_out[1], dL_dmix_out[2]);
		// }

		float dL_dfilter1_out[GABOR_HIDDEN_DIM] = {0};
		float dL_dfilter0_linear_out[GABOR_HIDDEN_DIM] = {0};
		for (int i = 0; i < GABOR_HIDDEN_DIM; i++) {
			dL_dfilter1_out[i] = dL_dmix_out[i] * filter_linear_out[0][i];
			dL_dfilter0_linear_out[i] = dL_dmix_out[i] * filter_out[1][i];
		}

		float dL_dinput[GABOR_IN_DIM] = {0};
		gabor_layers[1].backward(input, dL_dfilter1_out, dL_dinput, gabor_inter_vars[1]);

		float dL_dfilter0_out[GABOR_HIDDEN_DIM] = {0};
		linear_layers[0].backward(filter_out[0], dL_dfilter0_linear_out, dL_dfilter0_out);
		// if (debug) {
		// 	printf("back input %.8f %.8f, dL_dfilter0_linear_out: %.8f %.8f, filter_out0: %.8f %.8f, dL_dfilter0_out: %.8f %.8f\n", input[0], input[1], dL_dfilter0_linear_out[0], dL_dfilter0_linear_out[1], filter_out[0][0], filter_out[0][1], dL_dfilter0_out[0], dL_dfilter0_out[1]);
		// }

		gabor_layers[0].backward(input, dL_dfilter0_out, dL_input, gabor_inter_vars[0]);


		// float filter_out[GABOR_LAYER_NUM + 1][GABOR_HIDDEN_DIM];
		// GaborInterVars gabor_inter_vars[GABOR_LAYER_NUM + 1];
		// gabor_layers[0].forward(input, filter_out[0], gabor_inter_vars[0]);
		
		// if (debug) {
		// 	printf("back in: %.8f %.8f, filter_out: %.8f %.8f\n", input[0], input[1], filter_out[0][0], filter_out[0][1]);
		// 	gabor_inter_vars[0].print(input[0], input[1]);
		// }

		// float dL_dmix_out[GABOR_HIDDEN_DIM] = {0};
		// out_linear.backward(filter_out[0], dL_dcolor, dL_dmix_out);
		// gabor_layers[0].backward(input, dL_dmix_out, dL_input, gabor_inter_vars[0]);
		// gabor_layers[0].backward(input, dL_dcolor, dL_input, gabor_inter_vars[0]);

		// float tmp[GABOR_HIDDEN_DIM];
		// for (int j = 0; j < GABOR_HIDDEN_DIM; j++) {
		// 	tmp[j] = sinf(input[j]);
		// }
		// float dL_dcolor_tmp[GABOR_HIDDEN_DIM];
		// out_linear.backward(tmp, dL_dcolor, dL_dcolor_tmp);

		// for (int j = 0; j < GABOR_HIDDEN_DIM; j++) {
		// 	dL_input[j] = dL_dcolor_tmp[j] * cosf(input[j]);
		// }


	}

};

class Params {

private:
	int filters_stride = (GABOR_IN_DIM + 1) * GABOR_HIDDEN_DIM * 2 * (GABOR_LAYER_NUM + 1);
	int linears_stride = (GABOR_HIDDEN_DIM + 1) * GABOR_HIDDEN_DIM * GABOR_LAYER_NUM;
	int out_linear_stride = (GABOR_HIDDEN_DIM + 1) * GABOR_OUT_DIM;

public:
	float* filters = nullptr;
	float* linears = nullptr;
	float* out_linear = nullptr;

	float * dL_dfilters = nullptr;
	float * dL_dlinears = nullptr;
	float * dL_dout_linear = nullptr;

	Params() {}
	
    void set_params(
		const torch::Tensor& gabor_filters, const torch::Tensor& gabor_linears, const torch::Tensor& gabor_out_linear
	) {
		// contiguous() is used to ensure that the tensor is stored in a contiguous chunk of memory
		filters = gabor_filters.contiguous().data<float>(); // (N, x)
		linears = gabor_linears.contiguous().data<float>(); // (N, GABOR_HIDDEN_DIM + 1, GABOR_HIDDEN_DIM)
		out_linear = gabor_out_linear.contiguous().data<float>(); // (N, GABOR_HIDDEN_DIM + 1, GABOR_OUT_DIM)
	}

	void set_grads(
		torch::Tensor &dL_dgabor_filters, torch::Tensor &dL_dgabor_linears, torch::Tensor &dL_dgabor_out_linear 
	) {
		dL_dfilters = dL_dgabor_filters.contiguous().data<float>();
		dL_dlinears = dL_dgabor_linears.contiguous().data<float>();
		dL_dout_linear = dL_dgabor_out_linear.contiguous().data<float>();
	}

	__device__ void get_params(int idx, Network &net, bool get_grad) const {

		// l1_weight and l1_bias
		float * filters_ptr = filters + idx * filters_stride;
		for (int i = 0; i < GABOR_LAYER_NUM + 1; i++) {
			net.gabor_layers[i].mu = filters_ptr;
			filters_ptr += GABOR_IN_DIM * GABOR_HIDDEN_DIM;
			net.gabor_layers[i].gamma = filters_ptr;
			filters_ptr += GABOR_HIDDEN_DIM;
			net.gabor_layers[i].linear.weight = filters_ptr;
			filters_ptr += GABOR_HIDDEN_DIM * GABOR_IN_DIM;
			net.gabor_layers[i].linear.bias = filters_ptr;
			filters_ptr += GABOR_HIDDEN_DIM;
		}

		float * linears_ptr = linears + idx * linears_stride;
		for (int i = 0; i < GABOR_LAYER_NUM; i++) {
			net.linear_layers[i].weight = linears_ptr;
			linears_ptr += GABOR_HIDDEN_DIM * GABOR_HIDDEN_DIM;
			net.linear_layers[i].bias = linears_ptr;
			linears_ptr += GABOR_HIDDEN_DIM;
		}

		float * out_linear_ptr = out_linear + idx * out_linear_stride;
		net.out_linear.weight = out_linear_ptr;
		out_linear_ptr += GABOR_HIDDEN_DIM * GABOR_OUT_DIM;
		net.out_linear.bias = out_linear_ptr;

		if (get_grad) {
			
			float * dL_dfilters_ptr = dL_dfilters + idx * filters_stride;
			for (int i = 0; i < GABOR_LAYER_NUM + 1; i++) {
				net.gabor_layers[i].dmu = dL_dfilters_ptr;
				dL_dfilters_ptr += GABOR_IN_DIM * GABOR_HIDDEN_DIM;
				net.gabor_layers[i].dgamma = dL_dfilters_ptr;
				dL_dfilters_ptr += GABOR_HIDDEN_DIM;
				net.gabor_layers[i].linear.dweight = dL_dfilters_ptr;
				dL_dfilters_ptr += GABOR_HIDDEN_DIM * GABOR_IN_DIM;
				net.gabor_layers[i].linear.dbias = dL_dfilters_ptr;
				dL_dfilters_ptr += GABOR_HIDDEN_DIM;
			}

			float * dL_dlinears_ptr = dL_dlinears + idx * linears_stride;
			for (int i = 0; i < GABOR_LAYER_NUM; i++) {
				net.linear_layers[i].dweight = dL_dlinears_ptr;
				dL_dlinears_ptr += GABOR_HIDDEN_DIM * GABOR_HIDDEN_DIM;
				net.linear_layers[i].dbias = dL_dlinears_ptr;
				dL_dlinears_ptr += GABOR_HIDDEN_DIM;
			}

			float * dL_dout_linear_ptr = dL_dout_linear + idx * out_linear_stride;
			net.out_linear.dweight = dL_dout_linear_ptr;
			dL_dout_linear_ptr += GABOR_HIDDEN_DIM * GABOR_OUT_DIM;
			net.out_linear.dbias = dL_dout_linear_ptr;
		}
	}
};



#endif

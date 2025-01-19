#ifndef CUDA_RASTERIZER_NETWORK_H
#define CUDA_RASTERIZER_NETWORK_H

#include <torch/extension.h>
#include <glm/glm.hpp>
#include <cmath>

#define C_LAYER_NUM 1
#define C_IN_DIM 2
#define C_HIDDEN_DIM 8
#define C_OUT_DIM 3

#define A_LAYER_NUM 1
#define A_IN_DIM 2
#define A_HIDDEN_DIM 8
#define A_OUT_DIM 1

#define C_STRIDE (C_IN_DIM + 1) * C_HIDDEN_DIM * 2 * (C_LAYER_NUM + 1) + (C_HIDDEN_DIM + 1) * C_HIDDEN_DIM * C_LAYER_NUM + (C_HIDDEN_DIM + 1) * C_OUT_DIM
#define A_STRIDE (A_IN_DIM + 1) * A_HIDDEN_DIM * 2 * (A_LAYER_NUM + 1) + (A_HIDDEN_DIM + 1) * A_HIDDEN_DIM * A_LAYER_NUM + (A_HIDDEN_DIM + 1) * A_OUT_DIM


__forceinline__ __device__ float sigmoid(float x) {
	return 1.0f / (1.0f + expf(-x));
}

template<size_t dim>
__forceinline__ __device__ void sigmoid_forward(const float* in, float* out) {
	for (int i = 0; i < dim; i++) {
        out[i] = sigmoid(in[i]);
    }
}

template<size_t dim>
__forceinline__ __device__ void sigmoid_backward(const float* in, const float* grad_out, float* grad_in) {
    for (int i = 0; i < dim; i++) {
        float sigmoid_val = sigmoid(in[i]);
        grad_in[i] = grad_out[i] * sigmoid_val * (1.0f - sigmoid_val);
    }
}


template<size_t in_dim, size_t out_dim>
class Linear {
public:
	const float * weight, * bias;
	float * dweight, * dbias;

	__device__ Linear() {}
	__device__ Linear(const float * _weight, const float * _bias, float * _dweight, float * _dbias) {
		weight = _weight;
		bias = _bias;
		dweight = _dweight;
		dbias = _dbias;
	}
	
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

template<size_t hidden_dim>
class GaborInterVars {
public:
	float x2_sum = 0;
	float D[hidden_dim], linear_out[hidden_dim], sin_term[hidden_dim], exp_term[hidden_dim];

	__device__ GaborInterVars() {}
	__device__ void print(float u, float v) {
		printf("uv: %.3f %.3f, x2_sum: %.8f\n", u, v, x2_sum);
		for (int i = 0; i < hidden_dim; i++) {
			printf("uv: %.3f %.3f, D[%d]: %.8f, linear_out[%d]: %.8f, sin_term[%d]: %.8f, exp_term[%d]: %.8f\n", 
				u, v,
				i, D[i], i, linear_out[i], i, sin_term[i], i, exp_term[i]
			);
		}
	}
};


template<size_t in_dim, size_t hidden_dim>
class Gabor{
public:
	const float * mu, * gamma;
	float * dmu, * dgamma;
	Linear<in_dim, hidden_dim> linear;
	
	__device__ Gabor() {}
	__device__ Gabor(
		const float * _mu, const float * _gamma, const float * _linear_weight, const float * _linear_bias,
		float * _dmu, float * _dgamma, float * _dlinear_weight, float * _dlinear_bias
	) {
		mu = _mu;
		gamma = _gamma;
		dmu = _dmu;
		dgamma = _dgamma;
		linear = Linear<in_dim, hidden_dim>(_linear_weight, _linear_bias, _dlinear_weight, _dlinear_bias);
	}

	__device__ void forward(
		const float * input, float * output, GaborInterVars<hidden_dim> &inter_vars
	) const {
		
		inter_vars.x2_sum = 0;
		for (int i = 0; i < in_dim; i++) {
			inter_vars.x2_sum += input[i] * input[i];
		}

		for (int j = 0; j < hidden_dim; j++) {
			float mu_sum = 0, x_mu_sum = 0;
			for (int i = 0; i < in_dim; i++) {
				int offset = i * hidden_dim;
				mu_sum += mu[offset + j] * mu[offset + j];
				x_mu_sum += input[i] * mu[offset + j];
			}
			inter_vars.D[j] = inter_vars.x2_sum + mu_sum - 2 * x_mu_sum;
		}

		linear.forward(input, inter_vars.linear_out);

		for (int j = 0; j < hidden_dim; j++) {
			inter_vars.sin_term[j] = sinf(inter_vars.linear_out[j]);
			inter_vars.exp_term[j] = expf(-0.5f * inter_vars.D[j] * gamma[j]);
			output[j] = inter_vars.sin_term[j] * inter_vars.exp_term[j];
		}
	}

	__device__ void backward(
		const float * input, const float * dL_dout, float * dL_din, const GaborInterVars<hidden_dim> &inter_vars
	) {

		float dL_dD[hidden_dim] = {0};
		float dL_dlin[hidden_dim] = {0};
		for (int j = 0; j < hidden_dim; j++) {
			dL_dlin[j] = dL_dout[j] * inter_vars.exp_term[j] * cosf(inter_vars.linear_out[j]);
			dL_dD[j] = dL_dout[j] * inter_vars.sin_term[j] * inter_vars.exp_term[j] * (-0.5f * gamma[j]);
			atomicAdd(&dgamma[j], dL_dout[j] * inter_vars.sin_term[j] * inter_vars.exp_term[j] * (-0.5f * inter_vars.D[j]));
		}

		float local_din[in_dim] = {0};
		linear.backward(input, dL_dlin, local_din);

		for (int i = 0; i < in_dim; i++) dL_din[i] = 0;

		for (int j = 0; j < hidden_dim; j++) {
			for (int i = 0; i < in_dim; i++) {
				int off = i * hidden_dim + j;
				atomicAdd(&dmu[off], dL_dD[j] * 2.0f * (mu[off] - input[i]));
				dL_din[i] += dL_dD[j] * 2.0f * (input[i] - mu[off]);
			}
		}

		for (int i = 0; i < in_dim; i++) {
			dL_din[i] += local_din[i];
		}
	}

};


template<size_t layer_num, size_t in_dim, size_t hidden_dim, size_t out_dim>
class Network {
public:
	Gabor<in_dim, hidden_dim> gabor_layers[layer_num + 1];
	Linear<hidden_dim, hidden_dim> linear_layers[layer_num];
	Linear<hidden_dim, out_dim> out_linear;
	bool out_sig = false;

	__device__ Network(
		int idx, int total_stride, bool set_grads,
		float * params, float * grads, bool _out_sig = false, bool debug = false
	) {
		float * ptr = params + idx * total_stride;
		float * grad_ptr = grads;

		if (set_grads) {
			grad_ptr += idx * total_stride;
		}

		for (int i = 0; i < layer_num + 1; i++) {
			int gamma_offset = in_dim * hidden_dim;
			int linear_weight_offset = gamma_offset + hidden_dim;
			int linear_bias_offset = linear_weight_offset + hidden_dim * in_dim;

			if (set_grads) {
				gabor_layers[i] = Gabor<in_dim, hidden_dim>(
					ptr, ptr + gamma_offset, ptr + linear_weight_offset, ptr + linear_bias_offset,
					grad_ptr, grad_ptr + gamma_offset, grad_ptr + linear_weight_offset, grad_ptr + linear_bias_offset
				);
				grad_ptr += linear_bias_offset + hidden_dim;

			} else {
				gabor_layers[i] = Gabor<in_dim, hidden_dim>(
					ptr, ptr + gamma_offset, ptr + linear_weight_offset, ptr + linear_bias_offset,
					nullptr, nullptr, nullptr, nullptr
				);
			}
			
			ptr += linear_bias_offset + hidden_dim;
		}

		for (int i = 0; i < layer_num; i++) {
			int bias_offset = hidden_dim * hidden_dim;
			if (set_grads) {
				linear_layers[i] = Linear<hidden_dim, hidden_dim>(
					ptr, ptr + bias_offset, grad_ptr, grad_ptr + bias_offset
				);
				grad_ptr += bias_offset + hidden_dim;

			} else {
				linear_layers[i] = Linear<hidden_dim, hidden_dim>(
					ptr, ptr + bias_offset, nullptr, nullptr
				);
			}
			ptr += bias_offset + hidden_dim;
		}

		int bias_offset = hidden_dim * out_dim;
		if (set_grads) {
			out_linear = Linear<hidden_dim, out_dim>(
				ptr, ptr + bias_offset, grad_ptr, grad_ptr + bias_offset
			);
			grad_ptr += bias_offset + out_dim;
		} else {
			out_linear = Linear<hidden_dim, out_dim>(
				ptr, ptr + bias_offset, nullptr, nullptr
			);
		}
		ptr += bias_offset + out_dim;

		out_sig = _out_sig;
	}

	__device__ void forward(const float input[in_dim], float output[out_dim], bool debug) const {
		
		float filter_out[hidden_dim];
		GaborInterVars<hidden_dim> gabor_inter_vars;
		gabor_layers[0].forward(input, filter_out, gabor_inter_vars);

		float filter_linear_out[hidden_dim];
		for (int i = 1; i <= layer_num; i++) {
			linear_layers[i - 1].forward(filter_out, filter_linear_out);
	
			float filter1_out[hidden_dim];
			gabor_layers[i].forward(input, filter1_out, gabor_inter_vars);

			for (int j = 0; j < hidden_dim; j++) {
				filter_out[j] = filter1_out[j] * filter_linear_out[j];
			}

		}

		out_linear.forward(filter_out, output);
		if (out_sig) {
			sigmoid_forward<out_dim>(output, output);
		}

	}

	__device__ void forward_and_save_inter_vars(
		const float input[in_dim], float output[out_dim], 
		float filter_out[layer_num + 1][hidden_dim], float filter_linear_out[layer_num][hidden_dim], float mix_out[layer_num][hidden_dim], float final_out[out_dim],
		GaborInterVars<hidden_dim> gabor_inter_vars[layer_num + 1],
		bool debug = false
	) {
		gabor_layers[0].forward(input, filter_out[0], gabor_inter_vars[0]);
		if (debug) {
			// printf("filter_out[0], %.3f %.3f\n", filter_out[0][0], filter_out[0][1]);
			// printf("gabor_inter_vars[0] %.3f %.3f\n", gabor_inter_vars[0].x2_sum, gabor_inter_vars[0].D[0]);
			printf("filter_out[0]: ");
			for (int i = 0; i < hidden_dim; i++) printf("%.3f ", filter_out[0][i]);
			printf("\n");
			printf("gabor_inter_vars[0]: x2_sum %.3f\n", gabor_inter_vars[0].x2_sum);
			printf("gabor_inter_vars[0]: D: ");
			for (int i = 0; i < hidden_dim; i++) printf("%.3f ", gabor_inter_vars[0].D[i]);
			printf("\n");
			printf("gabor_inter_vars[0]: linear_out: ");
			for (int i = 0; i < hidden_dim; i++) printf("%.3f ", gabor_inter_vars[0].linear_out[i]);
			printf("\n");
			printf("gabor_inter_vars[0]: sin_term: ");
			for (int i = 0; i < hidden_dim; i++) printf("%.3f ", gabor_inter_vars[0].sin_term[i]);
			printf("\n");
			printf("gabor_inter_vars[0]: exp_term: ");
			for (int i = 0; i < hidden_dim; i++) printf("%.3f ", gabor_inter_vars[0].exp_term[i]);
			printf("\n");
		}

		linear_layers[0].forward(filter_out[0], filter_linear_out[0]);
		gabor_layers[1].forward(input, filter_out[1], gabor_inter_vars[1]);
			
		for (int j = 0; j < hidden_dim; j++) {
			mix_out[0][j] = filter_out[1][j] * filter_linear_out[0][j];
		}
		// if (debug) {
		// 	printf("filter_out[1], %.3f %.3f\n", filter_out[1][0], filter_out[1][1]);
		// 	printf("filter_linear_out[0], %.3f %.3f\n", filter_linear_out[0][0], filter_linear_out[0][1]);
		// 	printf("Gabor_inter_vars[1], %.3f %.3f\n", gabor_inter_vars[1].x2_sum, gabor_inter_vars[1].D[0]);
		// }

		out_linear.forward(mix_out[0], final_out);

		if (out_sig) {
			sigmoid_forward<out_dim>(final_out, output);
		} else {
			for (int i = 0; i < out_dim; i++) output[i] = final_out[i];
		}

		if (debug) {
			printf("filter_our[1]: ");
			for (int i = 0; i < hidden_dim; i++) printf("%.3f ", filter_out[1][i]);
			printf("\n");
			printf("filter_linear_out[0]: ");
			for (int i = 0; i < hidden_dim; i++) printf("%.3f ", filter_linear_out[0][i]);
			printf("\n");
			printf("mix_out[0]: ");
			for (int i = 0; i < hidden_dim; i++) printf("%.3f ", mix_out[0][i]);
			printf("\n");
			printf("final_out: ");
			for (int i = 0; i < out_dim; i++) printf("%.3f ", final_out[i]);
			printf("\n");

			printf("final_out address %p\n", final_out);
		}

	}

	__device__ void backward(
		const float * input, const float * dL_dout, float * dL_dinput, 
		float filter_out[][hidden_dim], float filter_linear_out[][hidden_dim], float mix_out[][hidden_dim], float final_out[out_dim],
		GaborInterVars<hidden_dim> gabor_inter_vars[layer_num + 1],
		bool debug
	) {
		
		if (debug) {
			printf("mix_out, %.3f %.3f dL_dout %.3f %.3f\n", mix_out[0][0],  mix_out[0][1],  dL_dout[0], dL_dout[1]);
		}
		
		float dL_doutact[out_dim] = {0};
		float dL_dmix_out[hidden_dim] = {0};
        float dL_dfilter_out[layer_num + 1][hidden_dim] = {0};
        float dL_dfilter_linear_out[layer_num][hidden_dim] = {0};
        float dL_dinput_gabor[layer_num + 1][in_dim] = {0};

		if (out_sig) {
			sigmoid_backward<out_dim>(mix_out[layer_num - 1], dL_dout, dL_doutact);
		} else {
			for (int i = 0; i < out_dim; i++) {
				dL_doutact[i] = dL_dout[i];
			}
		}
        
        out_linear.backward(mix_out[layer_num - 1], dL_doutact, dL_dmix_out);
		if (debug) {
			for (int i = 0; i < hidden_dim; i++) {
				printf("dL_dmix_out[%d], %f ", i, dL_dmix_out[i]);
			}
			printf("\n");
		}

        for (int i = layer_num - 1; i >= 0; i--) {
            for (int j = 0; j < hidden_dim; j++) {
                dL_dfilter_out[i + 1][j] = dL_dmix_out[j] * filter_linear_out[i][j];
                dL_dfilter_linear_out[i][j] = dL_dmix_out[j] * filter_out[i + 1][j];
            }
			if (debug) {
				for (int j = 0; j < hidden_dim; j++) {
					printf("filter_out[%d][%d] %f ", i + 1, j, filter_out[i + 1][j]);
					printf("filter_linear_out[%d][%d] %f ", i, j, filter_linear_out[i][j]);
					printf("dL_dfilter_out[%d][%d] %f ", i + 1, j, dL_dfilter_out[i + 1][j]);
					printf("dL_dfilter_linear_out[%d][%d] %f ", i, j, dL_dfilter_linear_out[i][j]);
				}
				printf("\n");
			}
            gabor_layers[i + 1].backward(input, dL_dfilter_out[i + 1], dL_dinput_gabor[i + 1], gabor_inter_vars[i + 1]);
            linear_layers[i].backward(filter_out[i], dL_dfilter_linear_out[i], dL_dfilter_out[i]);
        }
        gabor_layers[0].backward(input, dL_dfilter_out[0], dL_dinput_gabor[0], gabor_inter_vars[0]);

        for (int i = 0; i < in_dim; i++) {
            dL_dinput[i] = 0;
			
            for (int j = 0; j <= layer_num; j++) {
				if (debug) {
					printf("i,j %d %d, dL_dinput_gabor[j][i] %f\n", i,j, dL_dinput_gabor[j][i]);
				}
                dL_dinput[i] += dL_dinput_gabor[j][i];
            }
        }

		// float dL_dmix_out[hidden_dim] = {0};
		// out_linear.backward(mix_out[0], dL_dout, dL_dmix_out);

		// // printf("dL_dmix_out %.3f %.3f\n", dL_dmix_out[0], dL_dmix_out[1]);

		// float dL_dfilter1_out[hidden_dim] = {0};
		// float dL_dfilter0_linear_out[hidden_dim] = {0};
		// for (int i = 0; i < hidden_dim; i++) {
		// 	dL_dfilter1_out[i] = dL_dmix_out[i] * filter_linear_out[0][i];
		// 	dL_dfilter0_linear_out[i] = dL_dmix_out[i] * filter_out[1][i];
		// }

		// float dL_dinput_gabor_layer1[in_dim] = {0};
		// gabor_layers[1].backward(input, dL_dfilter1_out, dL_dinput_gabor_layer1, gabor_inter_vars[1]);

		// float dL_dfilter0_out[hidden_dim] = {0};
		// linear_layers[0].backward(filter_out[0], dL_dfilter0_linear_out, dL_dfilter0_out);

		// float dL_dinput_gabor_layer0[in_dim] = {0};
		// gabor_layers[0].backward(input, dL_dfilter0_out, dL_dinput_gabor_layer0, gabor_inter_vars[0]);

		// // printf("dL_dinput_gabor_layer1 %.3f %.3f dL_dinput_gabor_layer0 %.3f %.3f\n", dL_dinput_gabor_layer1[0], dL_dinput_gabor_layer1[1], dL_dinput_gabor_layer0[0], dL_dinput_gabor_layer0[1]);
		// for (int i = 0; i < in_dim; i++) {
		// 	dL_dinput[i] = dL_dinput_gabor_layer0[i] + dL_dinput_gabor_layer1[i];
		// }

	}

};

#endif

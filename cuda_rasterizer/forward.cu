/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec2 scale,
	float mod,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T,
	float3 &normal
) {

	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, mod);
	glm::mat3 L = R * S;

	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);

	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);

	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(
	glm::mat3 T, 
	float cutoff,
	float2& point_image,
	float2& extent
) {
	glm::vec3 t = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
	float d = glm::dot(t, T[2] * T[2]);
	if (d == 0.0) return false;
	glm::vec3 f = (1 / d) * t;

	glm::vec2 p = glm::vec2(
		glm::dot(f, T[0] * T[2]),
		glm::dot(f, T[1] * T[2])
	);

	glm::vec2 h0 = p * p - 
		glm::vec2(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1])
		);

	glm::vec2 h = sqrt(max(glm::vec2(1e-4, 1e-4), h0));
	point_image = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	
	// Compute transformation matrix
	glm::mat3 T;
	float3 normal;
	if (transMat_precomp == nullptr)
	{
		compute_transmat(((float3*)orig_points)[idx], scales[idx], scale_modifier, rotations[idx], projmatrix, viewmatrix, W, H, T, normal);
		float3 *T_ptr = (float3*)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	} else {
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0], 
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		normal = make_float3(0.0, 0.0, 1.0);
	}

#if DUAL_VISIABLE
	float cos = -sumf3(p_view * normal);
	if (cos == 0) return;
	float multiplier = cos > 0 ? 1: -1;
	normal = multiplier * normal;
#endif

#if TIGHTBBOX // no use in the paper, but it indeed help speeds.
	// the effective extent is now depended on the opacity of gaussian.
	float cutoff = sqrtf(max(9.f + 2.f * logf(opacities[idx]), 0.000001));
#else
	float cutoff = 3.0f;
#endif

	// Compute center and radius
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, cutoff, point_image, extent);
		if (!ok) return;
		radius = ceil(max(max(extent.x, extent.y), cutoff * FilterSize));
	}

	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Compute colors 
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	float* __restrict__ color_nets,
	float* __restrict__ alpha_nets,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others,
	bool neural_offset)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int round_size = BLOCK_SIZE;
	const int rounds = ((range.y - range.x + round_size - 1) / round_size);
	int toDo = range.y - range.x;

	// 378 %% 571
	// const bool print_debug = (pix.x == 208 && pix.y == 112) ? true: false;
	// const bool print_debug = (pix.x == 494 && (pix.y == 576)) ? true: false;
	const bool print_debug = false;
	// bool print_debug = true;
	// if (pix.x == 250)
	// printf("here x %d, y %d, print_debug %d\n", pix.x, pix.y, print_debug);

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[round_size];
	__shared__ float2 collected_xy[round_size];
	__shared__ float4 collected_normal_opacity[round_size];
	__shared__ float3 collected_Tu[round_size];
	__shared__ float3 collected_Tv[round_size];
	__shared__ float3 collected_Tw[round_size];
	// __shared__ Network collected_net[round_size];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[COLOR_CHANNELS] = { 0 };
	float color_net_res[C_OUT_DIM] = {0};
	float alpha_net_res[A_OUT_DIM] = {0};


#if RENDER_AXUTILITY
	// render axutility ouput
	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	// float median_weight = {0};
	float median_contributor = {-1};
	int pixel_depth_disorder = 0;

#endif
	Bucket<ForwardNode> bucket;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= round_size) {

		bool use_bucket_sort = true;
		if (i >= MAX_ROUNDS) {
			// if (print_debug)
				// printf("fw, pix (%d,%d) round %d, toDo %d\n", pix.x, pix.y, i, toDo);
			// break;
			use_bucket_sort = false;
		}
		// } else {
		// 	if (print_debug)
		// 	printf("fw, pix (%d,%d) round %d, toDo %d\n", pix.x, pix.y, i, toDo);
		// }
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == round_size)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		// int thread_rank = block.thread_rank();

		int progress = i * round_size + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();

		if (print_debug) printf("forward round %d, toDo %d, T %f\n", i, toDo, T);
		
		// Count the depth along the shooting ray
		float pixel_depth[BLOCK_SIZE] = {0};

		// Iterate over current batch
		for (int j = 0; !done && j < min(round_size, toDo); j++) {
			// printf("size of network %d\n", sizeof(Network));

			// printf("try to access net in forward.h renderCuda.\n");
			// printf("net: %p\n", net);
			// printf("l1_lw: %p\n", net->l1_lw);
			// printf("l1_lw[%d]: %f\n", 0, net->l1_lw[0]);
			
			// Keep track of current position in range
			contributor++;

			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			// Transform the two planes into local u-v system. 
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			// Cross product of two planes is a line, Eq. (9)
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			// Perspective division to get the intersection (u,v), Eq. (10)
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			// Add low pass filter
			float2 d = {pixf.x - xy.x, pixf.y - xy.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

			// float rho = min(rho3d, rho2d);
			float rho = rho3d;
			float2 uv = s;
			if (rho3d > rho2d) { 
				uv = {FilterInv * d.x, FilterInv * d.y};
				rho = rho2d;
			}

			// compute depth
			float depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
			// if a point is too small, its depth is not reliable?
			// depth = (rho3d <= rho2d) ? depth : Tw.z 
			if (depth < near_n) continue;

			float4 nor_o = collected_normal_opacity[j];
			// float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;
			float alpha;

			if (neural_offset) {
				bool debug_init = (pix.x == 250 && pix.y == 250)? true : false;
				Network<C_LAYER_NUM, C_IN_DIM, C_HIDDEN_DIM, C_OUT_DIM> color_net(
					collected_id[j], C_STRIDE, false,
					color_nets, nullptr, false, debug_init
				);
				// Network<A_LAYER_NUM, A_IN_DIM, A_HIDDEN_DIM, A_OUT_DIM> alpha_net(
				// 	collected_id[j], A_STRIDE, false,
				// 	alpha_nets, nullptr, true, debug_init
				// );
				
				// params->get_params(collected_id[j], net, false);
				const float uv_[] = {uv.x / 3, uv.y / 3};
				
				color_net.forward(uv_, color_net_res, false);
				// alpha_net.forward(uv_, alpha_net_res, false);
				// alpha = alpha_net_res[0];
				float power = -0.5f * rho;
				if (power > 0.0f)
					continue;
				alpha = opa * exp(power);

			} else {

				float power = -0.5f * rho;
				if (power > 0.0f)
					continue;

				// Eq. (2) from 3D Gaussian splatting paper.
				// Obtain alpha by multiplying with Gaussian opacity
				// and its exponential falloff from mean.
				// Avoid numerical instabilities (see paper appendix). 
				alpha = opa * exp(power);
			}
			
			alpha = min(0.99f, alpha);
			if (alpha < threshold_visible) continue;

			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Store the data in the bucket for sorting
			// // float color[COLOR_CHANNELS];
			// for (int ch = 0; ch < COLOR_CHANNELS; ch++) {
			// 	color[ch] = features[collected_id[j] * COLOR_CHANNELS + ch];
			// }

			// Keep track of last range entry to update this pixel.
			// last_contributor = contributor;
			last_contributor = i * round_size + j;
			
			ForwardNode node(j, depth, alpha);
			bucket.add(node);
			if (print_debug) {
				printf("fw add j %d, bucket_id %d\n", j, bucket.num-1);
				node.print();
			}
			// 	printf("fw add round %d, j %d, bucket_id %d alpha: %f, depth: %f, normal %f, color: %f\n", i, j, bucket.num-1, alpha, depth, normal[0], color[0]);
			// }
			
			if (!bucket.full()) continue;
			
			if (use_bucket_sort) {
				// Sort the bucket
				bucket.sort(true);
			}

			// Iterate over the sorted bucket
			for (int k = 0; k < bucket.num; k++) {

				// Fetch data from the bucket
				ForwardNode node = bucket.get(k);
				if (print_debug) {
					node.print();
				}

				float w = node.alpha * T;
#if RENDER_AXUTILITY
				// Render depth distortion map
				// Efficient implementation of distortion loss, see 2DGS' paper appendix.
				float A = 1-T;
				float m = far_n / (far_n - near_n) * (1 - near_n / node.depth);
				distortion += (m * m * A + M2 - 2 * m * M1) * w;
				D  += node.depth * w;
				M1 += m * w;
				M2 += m * m * w;
				// pixel_depth[j] = depth;

				if (T > 0.5) {
					median_depth = node.depth;
					// median_weight = w;
					median_contributor = i * round_size + node.local_j;
				}
				// Render normal map
				// for (int ch=0; ch<3; ch++) N[ch] += node.normal[ch] * w;
				N[0] += collected_normal_opacity[node.local_j].x * w;
				N[1] += collected_normal_opacity[node.local_j].y * w;
				N[2] += collected_normal_opacity[node.local_j].z * w;
#endif

				// Eq. (3) from 3D Gaussian splatting paper.
				int gauss_id = collected_id[node.local_j];
				for (int ch = 0; ch < COLOR_CHANNELS; ch++) C[ch] += features[gauss_id * COLOR_CHANNELS + ch] * w;
				
				T = T * (1 - node.alpha);
			}

			bucket.init();
			if (print_debug) {
				printf("bucket init\n");
			}
			

		}

		// Process the last batch in bucket
		if (bucket.num > 0) {

			if (use_bucket_sort) {
				// Sort the bucket
				bucket.sort(true);
			}

			// Iterate over the sorted bucket
			for (int k = 0; k < bucket.num; k++) {

				// Fetch data from the bucket
				ForwardNode node = bucket.get(k);
				if (print_debug) {
					printf("get k %d, bucket_id %d\n", k, bucket.num-1);
					node.print();
				}

				float w = node.alpha * T;
	#if RENDER_AXUTILITY
				// Render depth distortion map
				// Efficient implementation of distortion loss, see 2DGS' paper appendix.
				float A = 1-T;
				float m = far_n / (far_n - near_n) * (1 - near_n / node.depth);
				distortion += (m * m * A + M2 - 2 * m * M1) * w;
				D  += node.depth * w;
				M1 += m * w;
				M2 += m * m * w;
				// pixel_depth[j] = depth;

				if (T > 0.5) {
					median_depth = node.depth;
					// median_weight = w;
					median_contributor = i * round_size + node.local_j;
				}
				// Render normal map
				N[0] += collected_normal_opacity[node.local_j].x * w;
				N[1] += collected_normal_opacity[node.local_j].y * w;
				N[2] += collected_normal_opacity[node.local_j].z * w;
	#endif

				// Eq. (3) from 3D Gaussian splatting paper.
				int gauss_id = collected_id[node.local_j];
				for (int ch = 0; ch < COLOR_CHANNELS; ch++) C[ch] += features[gauss_id * COLOR_CHANNELS + ch] * w;
				
				T = T * (1 - node.alpha);
			}
		}

		if (inside && use_bucket_sort) {
			// last contributor number in the bucket
			n_contrib[pix_id + (2 + i) * H * W] = bucket.num > 0? bucket.num: BUCKET_SIZE;
			if (print_debug) {
				printf("assign n_contrib %d\n", n_contrib[pix_id + (2 + i) * H * W]);
			}
		}

		bucket.init();
		if (print_debug) {
			printf("bucket init. n_contrib %d\n", n_contrib[pix_id + (2 + i) * H * W]);
		}

		// for (int j = 1; j < bucket.num; j++) {
		// 	if (pixel_depth[j] < pixel_depth[j-1]) pixel_depth_disorder++;
		// }

	}

	

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		// if (pix.x == 250 && pix.y == 250) {
		// 	printf("fw, last contributor %d\n", last_contributor);
		// }
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < COLOR_CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		// if (print_debug) {
			// printf("fw, pix_id %d, bucket_num %d\n", pix_id, bucket.num);
		// }
#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		// out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
		out_others[pix_id + DISORDER_OFFSET * H * W] = float(pixel_depth_disorder);
		
#endif
	}

}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	float* color_nets,
	float* alpha_nets,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others,
	bool neural_offset)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0); // Query device 0
	// printf("Shared memory per SM: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
	// printf("Shared memory per block: %d KB\n", prop.sharedMemPerBlock / 1024);

	renderCUDA<COLOR_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		means2D,
		colors,
		transMats,
		depths,
		normal_opacity,
		color_nets, 
		alpha_nets,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others,
		neural_offset);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<COLOR_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}

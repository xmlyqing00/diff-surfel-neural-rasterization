#ifndef BUCKET_H
#define BUCKET_H

#include "auxiliary.h"

class Node {
public:
    float depth, alpha, normal[3], color[3];
    int contributor_id, next_id, gauss_id;
    public:
    __device__ Node() {
        this->next_id = -1;
        this->gauss_id = -1;
    }
    __device__ Node(int contributor_id, float depth, float alpha, float normal[3], float color[3]) {
        this->contributor_id = contributor_id;
        this->depth = depth;
        this->alpha = alpha;
        for (int i = 0; i < 3; i++) {
            this->normal[i] = normal[i];
            this->color[i] = color[i];
        }
        this->next_id = -1;
    }

    __device__ Node(int contributor_id, int gauss_id, float depth, float alpha, float normal[3], float color[3]) {
        this->contributor_id = contributor_id;
        this->gauss_id = gauss_id;
        this->depth = depth;
        this->alpha = alpha;
        for (int i = 0; i < 3; i++) {
            this->normal[i] = normal[i];
            this->color[i] = color[i];
        }
        this->next_id = -1;
    }
};

class Bucket {

public:
    Node nodes[BLOCK_SIZE];
    int heads[BLOCK_SIZE];
    // int tails[BLOCK_SIZE];
    int num;
    float depth_min, depth_max;
    int sorted[BLOCK_SIZE];
    bool sorted_flag;

    __device__ Bucket() {
        this->num = 0;
        this->depth_min = 1e9;
        this->depth_max = -1e9; 
        this->sorted_flag = false;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            this->heads[i] = -1;
            // this->tails[i] = -1;
        }
    }

    __device__ void sort(bool ascending=true, bool verbose=false) {
        float scale = depth_max - depth_min;
        int bucket_size = BLOCK_SIZE - 1;
        for (int i = 0; i < num; i++) {
            int bucket_idx = (int) ((nodes[i].depth - depth_min) / scale * bucket_size);
            
            if (heads[bucket_idx] == -1) {
                heads[bucket_idx] = i;
                // tails[bucket_idx] = i;
            } else {
                // nodes[tails[bucket_idx]].next_id = i;
                // tails[bucket_idx] = i;
                int node_idx = heads[bucket_idx];
                int prev_idx = -1;
                while (node_idx != -1 && nodes[node_idx].depth < nodes[i].depth) {
                    prev_idx = node_idx;
                    node_idx = nodes[node_idx].next_id;
                }
                if (prev_idx == -1) {
                    nodes[i].next_id = heads[bucket_idx];
                    heads[bucket_idx] = i;
                } else {
                    nodes[i].next_id = nodes[prev_idx].next_id;
                    nodes[prev_idx].next_id = i;
                }
            }
        }
        
        if (ascending) {
            for (int i = 0, j = 0; i < BLOCK_SIZE; i++) {
                int idx = heads[i];
                while (idx != -1) {
                    sorted[j++] = idx;
                    idx = nodes[idx].next_id;
                }
            }
        } else {
            for (int i = BLOCK_SIZE - 1, j = 0; i >= 0; i--) {
                int idx = heads[i];
                while (idx != -1) {
                    sorted[j++] = idx;
                    idx = nodes[idx].next_id;
                }
            }
        }
        
        sorted_flag = true;
    }

    __device__ void add(int contributor, float depth, float alpha, float normal[3], float color[3]) {
        nodes[num++] = Node(contributor, depth, alpha, normal, color);
        depth_min = fmin(depth_min, depth);
        depth_max = fmax(depth_max, depth);
    }

    __device__ void add(int contributor, int gauss_id, float depth, float alpha, float normal[3], float color[3]) {
        nodes[num++] = Node(contributor, gauss_id, depth, alpha, normal, color);
        depth_min = fmin(depth_min, depth);
        depth_max = fmax(depth_max, depth);
    }

    __device__ void get(int idx, int &contributor, float& depth, float& alpha, float normal[3], float color[3]) {
        idx = sorted[idx];
        contributor = nodes[idx].contributor_id;
        depth = nodes[idx].depth;
        alpha = nodes[idx].alpha;
        for (int i = 0; i < COLOR_CHANNELS; i++) {
            normal[i] = nodes[idx].normal[i];
            color[i] = nodes[idx].color[i];
        }
    }

    __device__ void get(int idx, int &contributor, int &gauss_id, float& depth, float& alpha, float normal[3], float color[3]) {
        idx = sorted[idx];
        contributor = nodes[idx].contributor_id;
        gauss_id = nodes[idx].gauss_id;
        depth = nodes[idx].depth;
        alpha = nodes[idx].alpha;
        for (int i = 0; i < COLOR_CHANNELS; i++) {
            normal[i] = nodes[idx].normal[i];
            color[i] = nodes[idx].color[i];
        }
    }

};

#endif
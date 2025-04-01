#ifndef BUCKET_H
#define BUCKET_H

#include <string>
#include <cstring>
#include "auxiliary.h"

#define BUCKET_SIZE 16

class Node {
public:
    float depth, alpha;
    int local_j;
    short next_id;
    __device__ Node() {}
    __device__ Node(int local_j, float depth, float alpha) {
        this->local_j = local_j;
        this->depth = depth;
        this->alpha = alpha;
        this->next_id = -1;
    }
};

template<typename T>
class Bucket {

private:
    short num;
    float depth_min, depth_max;

public:
    T nodes[BUCKET_SIZE];
    int heads[BUCKET_SIZE];
    int sorted[BUCKET_SIZE];
    bool sorted_flag;

    __device__ Bucket() {
        init();
    }

    __device__ void init() {
        this->num = 0;
        this->depth_min = 1e9;
        this->depth_max = -1e9; 
        this->sorted_flag = false;
        for (int i = 0; i < BUCKET_SIZE; i++) {
            this->heads[i] = -1;
        }
    }

    __device__ void sort(bool ascending=true) {
        // We don't sort if there is only zero or one element to avoid scale being zero error.
        if (num < 2) return;

        const float scale = depth_max - depth_min;
        for (short i = 0; i < num; i++) {
            short bucket_idx = (short) ((nodes[i].depth - depth_min) / scale * (BUCKET_SIZE - 1));
            // if (bucket_idx >= BUCKET_SIZE) {
            //     printf("bucket_idx %d, depth %f, depth_min %f, depth_max %f, scale %f, bucket_num %d\n", bucket_idx, nodes[i].depth, depth_min, depth_max, scale, num);
            //     assert(bucket_idx >= 0 && bucket_idx < BUCKET_SIZE);
            // }
            if (heads[bucket_idx] == -1) {
                heads[bucket_idx] = i;
            } else {
                short node_idx = heads[bucket_idx];
                short prev_idx = -1;
                if (ascending) {
                    while (node_idx != -1 && nodes[node_idx].depth < nodes[i].depth) {
                        prev_idx = node_idx;
                        node_idx = nodes[node_idx].next_id;
                    }
                } else {
                    while (node_idx != -1 && nodes[node_idx].depth > nodes[i].depth) {
                        prev_idx = node_idx;
                        node_idx = nodes[node_idx].next_id;
                    }
                }
                
                if (prev_idx == -1) {
                    nodes[i].next_id = heads[bucket_idx];
                    heads[bucket_idx] = i;
                } else {
                    nodes[i].next_id = node_idx;
                    nodes[prev_idx].next_id = i;
                }
            }
        }
        
        if (ascending) {
            for (short i = 0, j = 0; i < BUCKET_SIZE; i++) {
                short idx = heads[i];
                while (idx != -1) {
                    // assert(j < BUCKET_SIZE);
                    sorted[j++] = idx;
                    idx = nodes[idx].next_id;
                }
            }
        } else {
            for (short i = BUCKET_SIZE - 1, j = 0; i >= 0; i--) {
                short idx = heads[i];
                while (idx != -1) {
                    // assert(j < BUCKET_SIZE);
                    sorted[j++] = idx;
                    idx = nodes[idx].next_id;
                }
            }
        }
        
        sorted_flag = true;
    }

    __forceinline__ __device__ short size() {
        return num;
    }

    __forceinline__ __device__ bool full() {
        return num == BUCKET_SIZE;
    }

    __forceinline__ __device__ void push(const T &node) {
        nodes[num++] = node;
        depth_min = fmin(depth_min, node.depth);
        depth_max = fmax(depth_max, node.depth);
    }

    __forceinline__ __device__ T get(int idx) {
        if (sorted_flag) idx = sorted[idx];
        return nodes[idx];
    }

};

#endif
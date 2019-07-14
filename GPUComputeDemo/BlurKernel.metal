//
//  BlurKernel.metal
//  GPUComputeDemo
//
//  Created by liuRuiLong on 2019/7/11.
//  Copyright © 2019 Ray. All rights reserved.
//

#include <metal_stdlib>
#include <metal_common>

using namespace metal;

kernel void blur_kernel0(texture2d<float, access::sample> inTexture [[texture(0)]],
                        texture2d<float, access::write> outTexture [[texture(1)]],
                        const device int *radiusBuf [[buffer(0)]],
                        const device float *sumOfWeight [[buffer(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height()) {
        return;
    }
    
    int radius = *radiusBuf;
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_edge);

    if (radius == 0) {
        float4 input = inTexture.sample(sample, float2(gid.x, outTexture.get_height() - gid.y), 0.0);
        outTexture.write(input, gid.xy);
        return;
    }
    float sum_of_weight = *sumOfWeight;
    float m = 5;
    const int r_min = 0 - radius;
    const int r_max = 0 + radius + 1;

//    for (int i = r_min; i < r_max; ++i) {
//        for (int j = r_min; j < r_max; ++j) {
//            sum_of_weight += 1/((2 * 3.1415926) * pow(m, 2.0)) * exp(-((pow(i, 2.0) + pow(j, 2.0))/(2 * pow(m, 2.0))));
//        }
//    }
    
    
    float4 sum = 0.0;
    for (int i = r_min; i < r_max; ++i) {
        for (int j = r_min; j < r_max; ++j) {
            float weight = exp(-((pow(i, 2.0) + pow(j, 2.0))/(2 * pow(m, 2.0))))/(6.2831852 * pow(m, 2.0));
            float last_weight = weight / sum_of_weight;
            float4 input = inTexture.sample(sample, float2(gid.x + i, gid.y + j), 0.0);
            sum += input * last_weight;
        }
    }
    outTexture.write(sum, uint2(gid.x, outTexture.get_height() - gid.y));
}



kernel void blur_kernel(texture2d<float, access::sample> inTexture [[texture(0)]],
                        texture2d<float, access::write> outTexture [[texture(1)]],
                        const device int *radiusBuf [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height()) {
        return;
    }
    
    int radius = *radiusBuf;
    //采样器，用于从输入 Texture（对应着 opencl 中的 Image）获取输入位置的像素
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_edge);
    if (radius == 0) {
        float4 input = inTexture.sample(sample, float2(gid.x, gid.y), 0.0);
        outTexture.write(input, gid.xy);
        return;
    }
    
    float sum_of_weight = 0;
    float m = 5;
    const int r_min = 0 - radius;
    const int r_max = 0 + radius + 1;
    /*
     1. 对某像素点P周围特定半径内的像素点每一个点执行上述算法，得到最初的权重矩阵，σ 值可以调整
     2.1 对1中得到的权重矩阵所有权重进行求和，得到权重和
     */
    for (int i = r_min; i < r_max; ++i) {
        for (int j = r_min; j < r_max; ++j) {
            sum_of_weight += 1/((2 * 3.1415926) * pow(m, 2.0)) * exp(-((pow(i, 2.0) + pow(j, 2.0))/(2 * pow(m, 2.0))));
        }
    }
    
    float4 sum = 0.0;
    for (int i = r_min; i < r_max; ++i) {
        for (int j = r_min; j < r_max; ++j) {
            
            /*
             2.2 再使用每个点在 1 中得到的结果除以权重和，得到最终的权重矩阵
             */
            float weight = 1/((2 * 3.1415926) * pow(m, 2.0)) * exp(-((pow(i, 2.0) + pow(j, 2.0))/(2 * pow(m, 2.0))));
            
            float last_weight = weight / sum_of_weight;
            float4 input = inTexture.sample(sample, float2(gid.x + i, gid.y + j), 0.0);
            
            /*
             3. 对P点特定半径内像素点和 2 中得到的权重矩阵中对应位置的权重进行相乘，然后再相加，得到就是 P 点的高斯模糊值
             */
            sum += input * last_weight;
        }
    }
    outTexture.write(sum, uint2(gid.x, gid.y));
}

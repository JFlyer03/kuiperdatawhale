//
// Created by fss on 23-7-22.
//
#include "layer/abstract/layer_factory.hpp"
#include "../source/layer/details/convolution.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace kuiper_infer;

TEST(test_registry, create_layer_convforward) {
  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs(batch_size);

  const uint32_t in_channel = 2;
  for (uint32_t i = 0; i < batch_size; ++i) {
    sftensor input = std::make_shared<ftensor>(in_channel, 4, 4);
    input->data().slice(0) = "1,2,3,4;"
                             "5,6,7,8;"
                             "9,10,11,12;"
                             "13,14,15,16;";

    input->data().slice(1) = "1,2,3,4;"
                             "5,6,7,8;"
                             "9,10,11,12;"
                             "13,14,15,16;";
    inputs.at(i) = input;
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t kernel_count = 2;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->data().slice(0) = arma::fmat("1,2,3;"
                                         "3,2,1;"
                                         "1,2,3;");
    kernel->data().slice(1) = arma::fmat("1,2,3;"
                                         "3,2,1;"
                                         "1,2,3;");
    weights.push_back(kernel);
  }
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                              0, stride_h, stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs);
  outputs.at(0)->Show();
}

//
// Created by fss on 23-7-22.
//
#include "layer/abstract/layer_factory.hpp"
#include "../source/layer/details/convolution.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace kuiper_infer;

TEST(test_registry, create_layer_convGroupforward) {
    const uint32_t batch_size = 1;
    std::vector<sftensor> inputs(batch_size);
    std::vector<sftensor> outputs(batch_size);

    const uint32_t in_channel = 4;
    for (uint32_t i = 0; i < batch_size; ++i) {
        sftensor input = std::make_shared<ftensor>(in_channel, 4, 4);
        input->data().slice(0) = "1,2,3,4;"
                                 "5,6,7,8;"
                                 "9,10,11,12;"
                                 "13,14,15,16;";

        input->data().slice(1) = "1,2,3,4;"
                                 "5,6,7,8;"
                                 "9,10,11,12;"
                                 "13,14,15,16;";
        input->data().slice(2) = "1,2,3,4;"
                                 "5,6,7,8;"
                                 "9,10,11,12;"
                                 "13,14,15,16;";

        input->data().slice(3) = "1,2,3,4;"
                                 "5,6,7,8;"
                                 "9,10,11,12;"
                                 "13,14,15,16;";
        inputs.at(i) = input;
    }
    const uint32_t kernel_h = 3;
    const uint32_t kernel_w = 3;
    const uint32_t stride_h = 1;
    const uint32_t stride_w = 1;
    const uint32_t kernel_count = 2;
    std::vector<sftensor> weights;
    for (uint32_t i = 0; i < kernel_count; ++i) {
        // 确保分组里的 输入通道数与卷积核通道数一致
        // 原本输入通道数为4，分组数为2，所以每个分组的输入通道数为2；则每个卷积核的通道数也为2
        sftensor kernel = std::make_shared<ftensor>(2, kernel_h, kernel_w);
        kernel->data().slice(0) = arma::fmat("1,2,3;"
                                             "3,2,1;"
                                             "1,2,3;");
        kernel->data().slice(1) = arma::fmat("1,2,3;"
                                             "3,2,1;"
                                             "1,2,3;");
        weights.push_back(kernel);
    }
    ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                                0, stride_h, stride_w, 2, false);
    conv_layer.set_weights(weights);
    conv_layer.Forward(inputs, outputs);
    outputs.at(0)->Show();
}





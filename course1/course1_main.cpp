//
// Created by fss on 23-5-27.
//

#include <gtest/gtest.h>
#include <glog/logging.h>

int main(int argc, char *argv[]) {
  // 初始化Google Test框架
  testing::InitGoogleTest(&argc, argv);
  // 初始化Google Logging库
  google::InitGoogleLogging("Kuiper");
  // 设置日志输出目录和日志输出到标准输出
  FLAGS_log_dir = "../../course1/log";
  FLAGS_alsologtostderr = true;

  LOG(INFO) << "Start test...\n";
  return RUN_ALL_TESTS();
}
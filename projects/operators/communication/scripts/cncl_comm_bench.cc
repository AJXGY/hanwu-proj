/*
 * Purpose:
 *   Prototype CNRT/CNCL native communication benchmark for Cambricon MLU.
 *   This path bypasses PyTorch and directly tests AllReduce and Send/Recv
 *   style communication across multiple devices in one process.
 */

#include <cncl.h>
#include <cnrt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define CNRT_CHECK(cmd)                                                      \
  do {                                                                       \
    cnrtRet_t status = (cmd);                                                \
    if (status != CNRT_RET_SUCCESS) {                                        \
      throw std::runtime_error(std::string("CNRT failed: ") + #cmd);         \
    }                                                                        \
  } while (0)

#define CNCL_CHECK(cmd)                                                      \
  do {                                                                       \
    cnclResult_t status = (cmd);                                             \
    if (status != CNCL_RET_SUCCESS) {                                        \
      throw std::runtime_error(std::string("CNCL failed: ") + #cmd);         \
    }                                                                        \
  } while (0)

struct BenchConfig {
  std::string output_path = "results/processed/comm_bench_results.csv";
  std::vector<size_t> message_bytes = {
      1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};
  int warmup = 2;
  int iters = 5;
};

struct Stats {
  double avg_ms;
  double min_ms;
  double max_ms;
  double std_ms;
};

struct BenchContext {
  static constexpr int kNumComms = 2;
  int dev_list[kNumComms] = {0, 1};
  int rank_list[kNumComms] = {0, 1};
  cnclComm_t comms[kNumComms];
  cnrtQueue_t queues[kNumComms];
  void* send_buffers[kNumComms];
  void* recv_buffers[kNumComms];
  size_t max_bytes = 0;
};

std::vector<size_t> parse_sizes(const std::string& text) {
  std::vector<size_t> result;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      result.push_back(static_cast<size_t>(std::stoull(item)));
    }
  }
  return result;
}

BenchConfig parse_args(int argc, char** argv) {
  BenchConfig config;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--output" && i + 1 < argc) {
      config.output_path = argv[++i];
    } else if (arg == "--message-sizes" && i + 1 < argc) {
      config.message_bytes = parse_sizes(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      config.warmup = std::stoi(argv[++i]);
    } else if (arg == "--iters" && i + 1 < argc) {
      config.iters = std::stoi(argv[++i]);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + arg);
    }
  }
  return config;
}

BenchContext initialize_context(const BenchConfig& config) {
  BenchContext ctx;
  uint32_t num_dev = 0;
  CNRT_CHECK(cnrtGetDeviceCount(&num_dev));
  if (num_dev < 2) {
    throw std::runtime_error("Expected at least 2 MLU devices.");
  }

  ctx.max_bytes = *std::max_element(
      config.message_bytes.begin(), config.message_bytes.end());

  for (int i = 0; i < BenchContext::kNumComms; ++i) {
    CNRT_CHECK(cnrtSetDevice(ctx.dev_list[i]));
    CNRT_CHECK(cnrtQueueCreate(&ctx.queues[i]));
    CNRT_CHECK(cnrtMalloc(&ctx.send_buffers[i], ctx.max_bytes));
    CNRT_CHECK(cnrtMalloc(&ctx.recv_buffers[i], ctx.max_bytes));
    CNRT_CHECK(cnrtMemset(ctx.send_buffers[i], 1, ctx.max_bytes));
    CNRT_CHECK(cnrtMemset(ctx.recv_buffers[i], 0, ctx.max_bytes));
  }

  CNCL_CHECK(cnclInitComms(
      ctx.comms, BenchContext::kNumComms, ctx.dev_list, ctx.rank_list,
      BenchContext::kNumComms, nullptr));
  return ctx;
}

void finalize_context(BenchContext& ctx) {
  CNCL_CHECK(cnclDestroyComms(ctx.comms, BenchContext::kNumComms));
  for (int i = 0; i < BenchContext::kNumComms; ++i) {
    CNRT_CHECK(cnrtSetDevice(ctx.dev_list[i]));
    CNRT_CHECK(cnrtQueueDestroy(ctx.queues[i]));
    CNRT_CHECK(cnrtFree(ctx.send_buffers[i]));
    CNRT_CHECK(cnrtFree(ctx.recv_buffers[i]));
  }
}

void sync_queues(BenchContext& ctx) {
  for (int i = 0; i < BenchContext::kNumComms; ++i) {
    CNRT_CHECK(cnrtSetDevice(ctx.dev_list[i]));
    CNRT_CHECK(cnrtQueueSync(ctx.queues[i]));
  }
}

double run_allreduce(BenchContext& ctx, size_t bytes) {
  int count = static_cast<int>(bytes / sizeof(float));
  auto start = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  for (int i = 0; i < BenchContext::kNumComms; ++i) {
    threads.emplace_back([&, i]() {
      CNRT_CHECK(cnrtSetDevice(ctx.dev_list[i]));
      CNCL_CHECK(cnclAllReduce(
          ctx.send_buffers[i], ctx.recv_buffers[i], count, cnclFloat32,
          cnclSum, ctx.comms[i], ctx.queues[i]));
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  sync_queues(ctx);
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double run_send_recv_roundtrip(BenchContext& ctx, size_t bytes) {
  int count = static_cast<int>(bytes / sizeof(float));
  auto start = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  threads.emplace_back([&]() {
    CNRT_CHECK(cnrtSetDevice(ctx.dev_list[0]));
    CNCL_CHECK(cnclSend(
        ctx.send_buffers[0], count, cnclFloat32, 1, ctx.comms[0],
        ctx.queues[0]));
    CNCL_CHECK(cnclRecv(
        ctx.recv_buffers[0], count, cnclFloat32, 1, ctx.comms[0],
        ctx.queues[0]));
  });
  threads.emplace_back([&]() {
    CNRT_CHECK(cnrtSetDevice(ctx.dev_list[1]));
    CNCL_CHECK(cnclRecv(
        ctx.recv_buffers[1], count, cnclFloat32, 0, ctx.comms[1],
        ctx.queues[1]));
    CNCL_CHECK(cnclSend(
        ctx.send_buffers[1], count, cnclFloat32, 0, ctx.comms[1],
        ctx.queues[1]));
  });
  for (auto& thread : threads) {
    thread.join();
  }
  sync_queues(ctx);
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

Stats summarize(const std::vector<double>& values) {
  double total = 0.0;
  for (double value : values) {
    total += value;
  }
  double avg = total / static_cast<double>(values.size());
  double min_value = *std::min_element(values.begin(), values.end());
  double max_value = *std::max_element(values.begin(), values.end());
  double sq = 0.0;
  for (double value : values) {
    double delta = value - avg;
    sq += delta * delta;
  }
  double std = std::sqrt(sq / static_cast<double>(values.size()));
  return {avg, min_value, max_value, std};
}

Stats measure(
    BenchContext& ctx,
    const BenchConfig& config,
    double (*runner)(BenchContext&, size_t),
    size_t bytes) {
  for (int i = 0; i < config.warmup; ++i) {
    runner(ctx, bytes);
  }
  std::vector<double> values;
  for (int i = 0; i < config.iters; ++i) {
    values.push_back(runner(ctx, bytes));
  }
  return summarize(values);
}

void write_header(std::ofstream& out) {
  out << "operator,message_bytes,avg_ms,min_ms,max_ms,std_ms,world_size,device_type\n";
}

void write_row(
    std::ofstream& out,
    const std::string& op_name,
    size_t bytes,
    const Stats& stats) {
  out << op_name << "," << bytes << ","
      << std::fixed << std::setprecision(6)
      << stats.avg_ms << "," << stats.min_ms << "," << stats.max_ms << ","
      << stats.std_ms << ",2,MLU580\n";
}

int main(int argc, char** argv) {
  try {
    BenchConfig config = parse_args(argc, argv);
    BenchContext ctx = initialize_context(config);
    std::ofstream out(config.output_path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
      throw std::runtime_error("Failed to open output file.");
    }
    write_header(out);

    for (size_t bytes : config.message_bytes) {
      Stats stats = measure(ctx, config, run_allreduce, bytes);
      write_row(out, "all_reduce", bytes, stats);
      std::cout << "all_reduce " << bytes << " bytes avg_ms=" << stats.avg_ms
                << "\n";
    }
    for (size_t bytes : config.message_bytes) {
      Stats stats = measure(ctx, config, run_send_recv_roundtrip, bytes);
      write_row(out, "send_recv_roundtrip", bytes, stats);
      std::cout << "send_recv_roundtrip " << bytes
                << " bytes avg_ms=" << stats.avg_ms << "\n";
    }

    out.close();
    finalize_context(ctx);
    std::cout << "Wrote benchmark results to " << config.output_path << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 1;
  }
}

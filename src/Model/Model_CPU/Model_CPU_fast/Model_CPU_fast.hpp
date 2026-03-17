#pragma once

#include "../Model_CPU.hpp" // Ensure this path matches your project
#include <memory>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

constexpr std::size_t BLOCK_SIZE = 256; 

struct BlockTask {
    std::size_t I;
    std::size_t J;
};

struct BlockResult : BlockTask {
    std::vector<float> ax_I = std::vector<float>(BLOCK_SIZE, 0.0f);
    std::vector<float> ay_I = std::vector<float>(BLOCK_SIZE, 0.0f);
    std::vector<float> az_I = std::vector<float>(BLOCK_SIZE, 0.0f);

    std::vector<float> ax_J = std::vector<float>(BLOCK_SIZE, 0.0f);
    std::vector<float> ay_J = std::vector<float>(BLOCK_SIZE, 0.0f);
    std::vector<float> az_J = std::vector<float>(BLOCK_SIZE, 0.0f);
};

// --- FULL SCHEDULER DEFINITION GOES HERE ---
class ParallelScheduler {
private:
    std::vector<std::thread> workers;
    std::queue<BlockTask> task_queue;
    std::mutex task_mutex;
    std::condition_variable task_cv;
    
    std::queue<BlockResult> result_queue;
    std::mutex result_mutex;
    std::condition_variable result_cv;

    bool stop_workers = false;
    std::function<BlockResult(BlockTask)> compute_func;

    void worker_loop() {
        while (true) {
            BlockTask task;
            {
                std::unique_lock<std::mutex> lock(task_mutex);
                task_cv.wait(lock, [this] { return stop_workers || !task_queue.empty(); });
                if (stop_workers && task_queue.empty()) return;
                
                task = task_queue.front();
                task_queue.pop();
            }

            BlockResult result = compute_func(task);

            {
                std::lock_guard<std::mutex> lock(result_mutex);
                result_queue.push(std::move(result));
            }
            result_cv.notify_one();
        }
    }

public:
    ParallelScheduler(unsigned int num_threads, std::function<BlockResult(BlockTask)> compute_callback) 
        : compute_func(compute_callback) {
        for (unsigned int i = 0; i < num_threads; ++i) {
            workers.emplace_back(&ParallelScheduler::worker_loop, this);
        }
    }

    ~ParallelScheduler() {
        {
            std::lock_guard<std::mutex> lock(task_mutex);
            stop_workers = true;
        }
        task_cv.notify_all();
        for (auto& worker : workers) if (worker.joinable()) worker.join();
    }

    void submit_tasks(const std::vector<BlockTask>& tasks) {
        {
            std::lock_guard<std::mutex> lock(task_mutex);
            for (const auto& task : tasks) task_queue.push(task);
        }
        task_cv.notify_all();
    }

    BlockResult wait_and_pop_result() {
        std::unique_lock<std::mutex> lock(result_mutex);
        result_cv.wait(lock, [this] { return !result_queue.empty(); });
        BlockResult result = std::move(result_queue.front());
        result_queue.pop();
        return result;
    }
};

// --- NOW THE COMPILER KNOWS WHAT IT IS ---
class Model_CPU_fast : public Model_CPU
{
private:
    std::unique_ptr<ParallelScheduler> scheduler;
    
    BlockResult compute_block(const BlockTask& task);
    void apply_block_result(const BlockResult& result);

public:
    Model_CPU_fast(const Initstate& initstate, Particles& particles);
    virtual ~Model_CPU_fast() = default;

    virtual void step() override;
};
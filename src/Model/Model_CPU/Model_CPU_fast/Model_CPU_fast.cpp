#include <cmath>
#include <vector>
#include <atomic>
#include "Model_CPU_fast.hpp"
#include <xsimd/xsimd.hpp>
#include <omp.h>
#include <cstddef>
#include <iostream>
#include <chrono>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <algorithm>


class LapTimer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> last_mark;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

public:
    LapTimer() {
        start_time = std::chrono::high_resolution_clock::now();
        last_mark = start_time;
    }

    // Prints the time taken since the last lap
    void lap(const std::string& step_name) {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - last_mark).count();
        std::cout << "[Timer] " << step_name << " took " << duration << " microseconds.\n";
        last_mark = now; // Reset the mark for the next step
    }

    // Prints the total time since the timer was created
    void total() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time).count();
        std::cout << "[Timer] Total time: " << duration << " microseconds.\n";
    }
};

namespace xs = xsimd;
using batch_array = xs::batch<float>;

Model_CPU_fast::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
    unsigned int hardware_threads = std::thread::hardware_concurrency();
    unsigned int worker_threads = (hardware_threads > 1) ? hardware_threads - 1 : 1;
    
    scheduler = std::make_unique<ParallelScheduler>(
        worker_threads, 
        [this](BlockTask t) -> BlockResult { 
            // The routing happens here, outside the compute functions!
            if (t.is_diagonal) {
                return this->compute_block_diagonal(t);
            } else {
                return this->compute_block_off_diagonal(t);
            }
        }
    );
}

// The Apply Function (Main Thread)
void Model_CPU_fast::apply_block_result(const BlockResult& result) {
    std::size_t start_i = result.I * BLOCK_SIZE;
    std::size_t end_i = std::min<std::size_t>(start_i + BLOCK_SIZE, n_particles);
    std::size_t start_j = result.J * BLOCK_SIZE;
    std::size_t end_j = std::min<std::size_t>(start_j + BLOCK_SIZE, n_particles);

    for (std::size_t i = start_i; i < end_i; ++i) {
        std::size_t local_i = i - start_i;
        accelerationsx[i] += result.ax_I[local_i];
        accelerationsy[i] += result.ay_I[local_i];
        accelerationsz[i] += result.az_I[local_i];
    }

    for (std::size_t j = start_j; j < end_j; ++j) {
        std::size_t local_j = j - start_j;
        accelerationsx[j] += result.ax_J[local_j];
        accelerationsy[j] += result.ay_J[local_j];
        accelerationsz[j] += result.az_J[local_j];
    }
}

BlockResult Model_CPU_fast::compute_block_off_diagonal(const BlockTask& task) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::size_t simd_size = batch_array::size;
    BlockResult result;
    result.I = task.I;
    result.J = task.J;

    std::size_t start_i = task.I * BLOCK_SIZE;
    std::size_t end_i = std::min<std::size_t>(start_i + BLOCK_SIZE, n_particles);
    std::size_t start_j = task.J * BLOCK_SIZE;
    std::size_t end_j = std::min<std::size_t>(start_j + BLOCK_SIZE, n_particles);

    for (std::size_t i = start_i; i < end_i; i += simd_size) {
        if (i + simd_size > end_i) {
            for (std::size_t is = i; is < end_i; ++is) {
                float ax_i = 0, ay_i = 0, az_i = 0;
                for (std::size_t j = start_j; j < end_j; ++j) {
                    float dx = particles.x[j] - particles.x[is];
                    float dy = particles.y[j] - particles.y[is];
                    float dz = particles.z[j] - particles.z[is];
                    float r2 = dx*dx + dy*dy + dz*dz;
                    
                    float factor = (r2 < 1.0f) ? 10.0f : 10.0f / (r2 * std::sqrt(r2));
                    
                    ax_i += dx * factor * initstate.masses[j];
                    ay_i += dy * factor * initstate.masses[j];
                    az_i += dz * factor * initstate.masses[j];
                    
                    std::size_t local_j = j - start_j;
                    result.ax_J[local_j] -= dx * factor * initstate.masses[is];
                    result.ay_J[local_j] -= dy * factor * initstate.masses[is];
                    result.az_J[local_j] -= dz * factor * initstate.masses[is];
                }
                std::size_t local_is = is - start_i;
                result.ax_I[local_is] += ax_i;
                result.ay_I[local_is] += ay_i;
                result.az_I[local_is] += az_i;
            }
            break; 
        }

        auto pi_x = xs::load_unaligned(&particles.x[i]);
        auto pi_y = xs::load_unaligned(&particles.y[i]); 
        auto pi_z = xs::load_unaligned(&particles.z[i]);
        auto m_i  = xs::load_unaligned(&initstate.masses[i]);
        
        batch_array ax_i(0.f), ay_i(0.f), az_i(0.f);

        for (std::size_t j = start_j; j < end_j; ++j) {
            batch_array pj_x(particles.x[j]);
            batch_array pj_y(particles.y[j]);
            batch_array pj_z(particles.z[j]);

            auto dx = pj_x - pi_x;
            auto dy = pj_y - pi_y;
            auto dz = pj_z - pi_z;

            auto r2 = xs::fma(dx, dx, xs::fma(dy, dy, dz * dz));
            auto r2_safe = xs::select(r2 < 1e-6f, batch_array(1.0f), r2);
            auto r = xs::rsqrt(r2_safe);
            
            auto factor_base = xs::select(r2 < 1.0f, batch_array(10.0f), 10.0f * r * r * r);

            auto factor_i = factor_base * initstate.masses[j];
            ax_i = xs::fma(dx, factor_i, ax_i);
            ay_i = xs::fma(dy, factor_i, ay_i);
            az_i = xs::fma(dz, factor_i, az_i);

            auto factor_j = factor_base * m_i;
            std::size_t local_j = j - start_j;
            result.ax_J[local_j] += xs::hadd(-dx * factor_j);
            result.ay_J[local_j] += xs::hadd(-dy * factor_j);
            result.az_J[local_j] += xs::hadd(-dz * factor_j);
        }

        std::size_t local_i = i - start_i;

        // 1. Load the existing values from the result array directly into SIMD registers
        auto res_ax = xs::load_unaligned(&result.ax_I[local_i]);
        auto res_ay = xs::load_unaligned(&result.ay_I[local_i]);
        auto res_az = xs::load_unaligned(&result.az_I[local_i]);

        // 2. Perform purely vectorized SIMD addition
        res_ax += ax_i;
        res_ay += ay_i;
        res_az += az_i;

        // 3. Store the updated vectors straight back into the array
        res_ax.store_unaligned(&result.ax_I[local_i]);
        res_ay.store_unaligned(&result.ay_I[local_i]);
        res_az.store_unaligned(&result.az_I[local_i]);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    total_worker_compute_time.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count(), std::memory_order_relaxed);
    return result;
}

BlockResult Model_CPU_fast::compute_block_diagonal(const BlockTask& task) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::size_t simd_size = batch_array::size;
    BlockResult result;
    result.I = task.I;
    result.J = task.J;

    std::size_t start_i = task.I * BLOCK_SIZE;
    std::size_t end_i = std::min<std::size_t>(start_i + BLOCK_SIZE, n_particles);
    std::size_t start_j = task.J * BLOCK_SIZE;
    std::size_t end_j = std::min<std::size_t>(start_j + BLOCK_SIZE, n_particles);

    alignas(64) float offsets[batch_array::size];
    for (std::size_t k = 0; k < batch_array::size; ++k) offsets[k] = static_cast<float>(k);
    batch_array lane_offsets = xs::load_aligned(offsets);

    for (std::size_t i = start_i; i < end_i; i += simd_size) {
        if (i + simd_size > end_i) {
            for (std::size_t is = i; is < end_i; ++is) {
                float ax_i = 0, ay_i = 0, az_i = 0;
                for (std::size_t j = start_j; j < end_j; ++j) {
                    if (j <= is) continue; 

                    float dx = particles.x[j] - particles.x[is];
                    float dy = particles.y[j] - particles.y[is];
                    float dz = particles.z[j] - particles.z[is];
                    float r2 = dx*dx + dy*dy + dz*dz;
                    
                    float factor = (r2 < 1.0f) ? 10.0f : 10.0f / (r2 * std::sqrt(r2));
                    
                    ax_i += dx * factor * initstate.masses[j];
                    ay_i += dy * factor * initstate.masses[j];
                    az_i += dz * factor * initstate.masses[j];
                    
                    std::size_t local_j = j - start_j;
                    result.ax_J[local_j] -= dx * factor * initstate.masses[is];
                    result.ay_J[local_j] -= dy * factor * initstate.masses[is];
                    result.az_J[local_j] -= dz * factor * initstate.masses[is];
                }
                std::size_t local_is = is - start_i;
                result.ax_I[local_is] += ax_i;
                result.ay_I[local_is] += ay_i;
                result.az_I[local_is] += az_i;
            }
            break; 
        }

        auto pi_x = xs::load_unaligned(&particles.x[i]);
        auto pi_y = xs::load_unaligned(&particles.y[i]); 
        auto pi_z = xs::load_unaligned(&particles.z[i]);
        auto m_i  = xs::load_unaligned(&initstate.masses[i]);
        
        batch_array i_indices = lane_offsets + static_cast<float>(i);
        batch_array ax_i(0.f), ay_i(0.f), az_i(0.f);

        for (std::size_t j = start_j; j < end_j; ++j) {
            batch_array pj_x(particles.x[j]);
            batch_array pj_y(particles.y[j]);
            batch_array pj_z(particles.z[j]);

            auto dx = pj_x - pi_x;
            auto dy = pj_y - pi_y;
            auto dz = pj_z - pi_z;

            auto r2 = xs::fma(dx, dx, xs::fma(dy, dy, dz * dz));
            auto r = xs::rsqrt(r2);
            auto factor_base = xs::select(r2 < 1.0f, batch_array(10.0f), 10.0f * r * r * r);

            // MASKING
            auto active_mask = batch_array(static_cast<float>(j)) > i_indices;
            factor_base = xs::select(active_mask, factor_base, batch_array(0.0f));

            auto factor_i = factor_base * initstate.masses[j];
            ax_i = xs::fma(dx, factor_i, ax_i);
            ay_i = xs::fma(dy, factor_i, ay_i);
            az_i = xs::fma(dz, factor_i, az_i);

            auto factor_j = factor_base * m_i;
            std::size_t local_j = j - start_j;
            result.ax_J[local_j] += xs::hadd(-dx * factor_j);
            result.ay_J[local_j] += xs::hadd(-dy * factor_j);
            result.az_J[local_j] += xs::hadd(-dz * factor_j);
        }

        std::size_t local_i = i - start_i;
        auto res_ax = xs::load_unaligned(&result.ax_I[local_i]);
        auto res_ay = xs::load_unaligned(&result.ay_I[local_i]);
        auto res_az = xs::load_unaligned(&result.az_I[local_i]);
        res_ax += ax_i;
        res_ay += ay_i;
        res_az += az_i;
        res_ax.store_unaligned(&result.ax_I[local_i]);
        res_ay.store_unaligned(&result.ay_I[local_i]);
        res_az.store_unaligned(&result.az_I[local_i]);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    total_worker_compute_time.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count(), std::memory_order_relaxed);
    return result;
}

void Model_CPU_fast::step()
{
    // Start the clock!
    LapTimer timer; 
    // --- LAP 1: Initialization ---
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0.0f);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0.0f);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0.0f);
    
    timer.lap("Zeroing arrays");

    // --- LAP 2: Task Generation and Submission ---

    std::size_t simd_size = batch_array::size;
    std::size_t num_blocks = (n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::vector<BlockTask> tasks;
    for (std::size_t I = 0; I < num_blocks; ++I) {
        for (std::size_t J = 0; J <= I; ++J) {
            // Add the flag here: True if diagonal, False if off-diagonal
            tasks.push_back({I, J, I == J});
        }
    }

    scheduler->submit_tasks(tasks);
    
    timer.lap("Task generation & submission");

    // --- LAP 3: Result Processing (The heavy lifting) ---
    std::size_t tasks_completed = 0;
    std::size_t total_tasks = tasks.size();

    long long total_wait_time = 0;
    long long total_apply_time = 0;

    while (tasks_completed < total_tasks) {
        // Measure how long the main thread spends WAITING for a worker
        auto t1 = std::chrono::high_resolution_clock::now();
        BlockResult result = scheduler->wait_and_pop_result(); 
        
        // Measure how long the main thread spends APPLYING the data
        auto t2 = std::chrono::high_resolution_clock::now();
        apply_block_result(result);
        
        auto t3 = std::chrono::high_resolution_clock::now();

        total_wait_time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        total_apply_time += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

        tasks_completed++;
    }

    std::cout << "[Timer] Time spent WAITING for workers: " << total_wait_time << " us\n";
    std::cout << "[Timer] Time spent APPLYING results: " << total_apply_time << " us\n";
    timer.lap("Parallel computation & applying results (Total)");
    
    // --- PRINT INTERNAL SCHEDULER STATS ---
    long long pure_math_time = total_worker_compute_time.load();
    
    // Reset it for the next frame
    total_worker_compute_time.store(0); 

    std::cout << "[Scheduler Stats] Total pure math time across ALL threads: " << pure_math_time << " us\n";
    
    // If you have 15 worker threads, the "wall time" they spent doing math is pure_math / 15.
    unsigned int worker_count = std::thread::hardware_concurrency() - 1;
    long long expected_wall_time = pure_math_time / (worker_count > 0 ? worker_count : 1);
    
    std::cout << "[Scheduler Stats] Expected wall time (Math / Threads): ~" << expected_wall_time << " us\n";
    std::cout << "[Scheduler Stats] Actual total step time: " << 70691 << " us\n"; // Using your number for reference
    
    timer.total();
    // --- LAP 4: Final Integration ---
    batch_array v_factor(2.0f);
    batch_array p_factor(0.1f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_particles; i += simd_size) {
        if (i + simd_size > n_particles) {
            for (size_t k = i; k < n_particles; ++k) {
                velocitiesx[k] += accelerationsx[k] * 2.0f;
                velocitiesy[k] += accelerationsy[k] * 2.0f;
                velocitiesz[k] += accelerationsz[k] * 2.0f;
                particles.x[k] += velocitiesx[k] * 0.1f;
                particles.y[k] += velocitiesy[k] * 0.1f;
                particles.z[k] += velocitiesz[k] * 0.1f;
            }
            continue;
        }

        auto ax = xs::load_unaligned(&accelerationsx[i]);
        auto ay = xs::load_unaligned(&accelerationsy[i]);
        auto az = xs::load_unaligned(&accelerationsz[i]);

        auto vx = xs::load_unaligned(&velocitiesx[i]);
        auto vy = xs::load_unaligned(&velocitiesy[i]);
        auto vz = xs::load_unaligned(&velocitiesz[i]);

        vx = xs::fma(ax, v_factor, vx);
        vy = xs::fma(ay, v_factor, vy);
        vz = xs::fma(az, v_factor, vz);

        vx.store_unaligned(&velocitiesx[i]);
        vy.store_unaligned(&velocitiesy[i]);
        vz.store_unaligned(&velocitiesz[i]);

        auto px = xs::load_unaligned(&particles.x[i]);
        auto py = xs::load_unaligned(&particles.y[i]);
        auto pz = xs::load_unaligned(&particles.z[i]);

        px = xs::fma(vx, p_factor, px);
        py = xs::fma(vy, p_factor, py);
        pz = xs::fma(vz, p_factor, pz);

        px.store_unaligned(&particles.x[i]);
        py.store_unaligned(&particles.y[i]);
        pz.store_unaligned(&particles.z[i]);
    }

    timer.lap("OpenMP integration");
    // Print the total time for the step
    timer.total();
}
#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using batch_array = xs::batch<float>;

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

    std::size_t simd_size = batch_array::size;
    // Parallelize the outer loop over particles 'i', no depandence between the i
    #pragma omp parallel for schedule(dynamic) //dynamic seems better on my computer
    for (std::size_t i = 0; i < n_particles; ++i) {
        
        // Load position of particle i as they are used for each j
        float pi_x = particles.x[i];
        float pi_y = particles.y[i];
        float pi_z = particles.z[i];
        
        batch_array ax_sum(0.f);
        batch_array ay_sum(0.f);
        batch_array az_sum(0.f);

        std::size_t j = 0;
        
        for (; j < n_particles - simd_size + 1; j += simd_size) {
            
            // Load batch of j positions, it's unaligned as even if particules struct are store consecutively there is.x.y.z so all .x are not consecutive
            auto pj_x = xs::load_unaligned(&particles.x[j]);
            auto pj_y = xs::load_unaligned(&particles.y[j]);
            auto pj_z = xs::load_unaligned(&particles.z[j]);

            // Calculate distance vector (pj - pi)
            auto dx = pj_x - pi_x;
            auto dy = pj_y - pi_y;
            auto dz = pj_z - pi_z;

            // r^2 = dx*dx + dy*dy + dz*dz
            auto r2 = xs::fma(dx, dx, xs::fma(dy, dy, dz * dz));

            // Check condition: r^2 < 1.0
            // logic: if (r2 < 1.0) val = 10.0; else val = 10.0 / (sqrt(r2)^3)
            auto mask = r2 < 1.0f;

            // Calculate "else" branch
            auto r = xs::rsqrt(r2);
            auto val_far = 10.0f * r * r * r;
            
            // Select value based on mask
            // If i == j, r2 is 0. Mask is true. factor becomes 10.0
            // Force contribution = dx * 10.0 idem for dy, dz. Since dx,dy,dz is 0, force is 0
            auto factor = xs::select(mask, batch_array(10.0f), val_far);
            auto m_j  = xs::load_unaligned(&initstate.masses[j]);
            factor *= m_j;

            //No concurrent access it's safe there as only one thread for each i and sum over j
            ax_sum = xs::fma(dx, factor, ax_sum);
            ay_sum = xs::fma(dy, factor, ay_sum);
            az_sum = xs::fma(dz, factor, az_sum);
        }

        // Horizontal reduction: Sum the vector lanes into a single scalar for particle i
        accelerationsx[i] += xs::hadd(ax_sum);
        accelerationsy[i] += xs::hadd(ay_sum);
        accelerationsz[i] += xs::hadd(az_sum);

        // Handle remaining j particles
        for (; j < n_particles; ++j) {
            if (i != j) {
                float dx = particles.x[j] - pi_x;
                float dy = particles.y[j] - pi_y;
                float dz = particles.z[j] - pi_z;
                
                float r2 = dx*dx + dy*dy + dz*dz;
                float dij;
                
                if (r2 < 1.0f) {
                    dij = 10.0f;
                } else {
                    float r = std::sqrt(r2);
                    dij = 10.0f / (r * r2);
                }

                accelerationsx[i] += dx * dij * initstate.masses[j];
                accelerationsy[i] += dy * dij * initstate.masses[j];
                accelerationsz[i] += dz * dij * initstate.masses[j];
            }
        }
    }

    batch_array v_factor(2.0f);
    batch_array p_factor(0.1f);

    std::size_t i = 0;
    #pragma omp parallel for private(i)
    for (i = 0; i < n_particles; i += simd_size) {
        // Ensure we don't read/write past bounds in the last iteration
        if (i + simd_size > n_particles) {
             // Fallback to scalar for the very last chunk if needed
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

        // Load Accelerations
        auto ax = xs::load_unaligned(&accelerationsx[i]);
        auto ay = xs::load_unaligned(&accelerationsy[i]);
        auto az = xs::load_unaligned(&accelerationsz[i]);

        // Load Velocities
        auto vx = xs::load_unaligned(&velocitiesx[i]);
        auto vy = xs::load_unaligned(&velocitiesy[i]);
        auto vz = xs::load_unaligned(&velocitiesz[i]);

        // Update Velocities: v += a * 2.0
        vx = xs::fma(ax, v_factor, vx);
        vy = xs::fma(ay, v_factor, vy);
        vz = xs::fma(az, v_factor, vz);

        // Store Velocities back
        vx.store_unaligned(&velocitiesx[i]);
        vy.store_unaligned(&velocitiesy[i]);
        vz.store_unaligned(&velocitiesz[i]);

        // Load Positions
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
}
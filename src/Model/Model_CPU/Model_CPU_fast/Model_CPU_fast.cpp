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
    std::size_t simd_size = batch_array::size;    
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i <= n_particles - simd_size; i += simd_size) {

        // Load position of particles i to i+(simd_size-1)
        auto pi_x = xs::load_unaligned(&particles.x[i]);
        auto pi_y = xs::load_unaligned(&particles.y[i]); 
        auto pi_z = xs::load_unaligned(&particles.z[i]); 
        
        batch_array ax_i_i4(0.f);
        batch_array ay_i_i4(0.f);
        batch_array az_i_i4(0.f);
        
        // Loop over ALL j particles. Step by 1.
        for (std::size_t j = 0; j < n_particles; j++) {
            
            // FIX: Broadcast the single j particle properties to all SIMD lanes
            // This creates a batch like: [x_j, x_j, x_j, x_j]
            batch_array pj_x(particles.x[j]);
            batch_array pj_y(particles.y[j]);
            batch_array pj_z(particles.z[j]);

            auto dx = pj_x - pi_x;
            auto dy = pj_y - pi_y;
            auto dz = pj_z - pi_z;

            // r^2 = dx*dx + dy*dy + dz*dz
            auto r2 = xs::fma(dx, dx, xs::fma(dy, dy, dz * dz));

            // Check condition: r^2 < 1.0f
            auto mask = r2 < 1.0f;

            // Calculate "else" branch
            // Note: r * r * r is faster than r^3
            auto r = xs::rsqrt(r2);
            auto val_far = 10.0f * r * r * r;
            
            auto factor = xs::select(mask, batch_array(10.0f), val_far);

            batch_array m_j(initstate.masses[j]); 
            factor *= m_j;

            // Accumulate accelerations
            ax_i_i4 = xs::fma(dx, factor, ax_i_i4);
            ay_i_i4 = xs::fma(dy, factor, ay_i_i4);
            az_i_i4 = xs::fma(dz, factor, az_i_i4);
        }

        // Store results
        ax_i_i4.store_unaligned(&accelerationsx[i]);
        ay_i_i4.store_unaligned(&accelerationsy[i]);
        az_i_i4.store_unaligned(&accelerationsz[i]);
    }

    batch_array v_factor(2.0f);
    batch_array p_factor(0.1f);

    std::size_t j = 0;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_particles; i += simd_size) {
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
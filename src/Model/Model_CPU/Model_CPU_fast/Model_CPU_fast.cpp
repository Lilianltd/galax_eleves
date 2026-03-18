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
    std::size_t double_simd = 2 * simd_size;

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i <= n_particles - double_simd; i += double_simd) {
        auto pi_x_1 = xs::load_unaligned(&particles.x[i]);
        auto pi_y_1 = xs::load_unaligned(&particles.y[i]); 
        auto pi_z_1 = xs::load_unaligned(&particles.z[i]); 
        
        batch_array ax_1(0.f), ay_1(0.f), az_1(0.f);
        
        for (std::size_t j = 0; j < n_particles; ++j) {
            batch_array pj_x(particles.x[j]);
            batch_array pj_y(particles.y[j]);
            batch_array pj_z(particles.z[j]);

            auto dx = pj_x - pi_x_1;
            auto dy = pj_y - pi_y_1;
            auto dz = pj_z - pi_z_1;

            auto r2 = xs::fma(dx, dx, xs::fma(dy, dy, dz * dz));
            auto mask = r2 < 1.0f;
            auto r = xs::rsqrt(r2);
            auto val_far = 10.0f * r * r * r;
            auto factor = xs::select(mask, batch_array(10.0f), val_far);

            batch_array m_j(initstate.masses[j]); 
            factor *= m_j;

            ax_1 = xs::fma(dx, factor, ax_1);
            ay_1 = xs::fma(dy, factor, ay_1);
            az_1 = xs::fma(dz, factor, az_1);
        }

        ax_1.store_unaligned(&accelerationsx[i]);
        ay_1.store_unaligned(&accelerationsy[i]);
        az_1.store_unaligned(&accelerationsz[i]);

        std::size_t i_next = i + simd_size;
        auto pi_x_2 = xs::load_unaligned(&particles.x[i_next]);
        auto pi_y_2 = xs::load_unaligned(&particles.y[i_next]); 
        auto pi_z_2 = xs::load_unaligned(&particles.z[i_next]); 
        
        batch_array ax_2(0.f), ay_2(0.f), az_2(0.f);
        
        for (std::size_t j = n_particles; j-- > 0; ) {
            batch_array pj_x(particles.x[j]);
            batch_array pj_y(particles.y[j]);
            batch_array pj_z(particles.z[j]);

            auto dx = pj_x - pi_x_2;
            auto dy = pj_y - pi_y_2;
            auto dz = pj_z - pi_z_2;

            auto r2 = xs::fma(dx, dx, xs::fma(dy, dy, dz * dz));
            auto mask = r2 < 1.0f;
            auto r = xs::rsqrt(r2);
            auto val_far = 10.0f * r * r * r;
            auto factor = xs::select(mask, batch_array(10.0f), val_far);

            batch_array m_j(initstate.masses[j]); 
            factor *= m_j;

            ax_2 = xs::fma(dx, factor, ax_2);
            ay_2 = xs::fma(dy, factor, ay_2);
            az_2 = xs::fma(dz, factor, az_2);
        }

        ax_2.store_unaligned(&accelerationsx[i_next]);
        ay_2.store_unaligned(&accelerationsy[i_next]);
        az_2.store_unaligned(&accelerationsz[i_next]);
    }
    
    batch_array v_factor(2.0f);
    batch_array p_factor(0.1f);

    std::size_t j = 0;
    #pragma omp parallel for schedule(static)
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
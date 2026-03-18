
#define MODEL_CPU_FAST_HPP_

#include "../Model_CPU.hpp"

class Model_CPU_fast : public Model_CPU
{
public:
    Model_CPU_fast(const Initstate& initstate, Particles& particles);

    virtual ~Model_CPU_fast() = default;

    virtual void step();
};

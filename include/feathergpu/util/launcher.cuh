#pragma once

template <typename... Arguments>
__host__ void feather_cpu_launcher( void(*f)(Arguments...), Arguments... args)
{
    f(args...);
}

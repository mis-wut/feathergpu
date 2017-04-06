#pragma once
inline int if_debug()
{
   char* GPU_DEBUG;
   GPU_DEBUG = getenv ("GPU_DEBUG");

   if (GPU_DEBUG != NULL)
       if (atoi(GPU_DEBUG) > 0) return 1;
   return 0;
}

inline int getenv_extract_int(const char *env_name, int min, int max, int default_value)
{
   char* env;
   env = getenv (env_name);
   int ret = default_value;

   if (env != NULL){
       ret = atoi(env);
       if (ret < min) ret = min;
       if (ret > max) ret = max;
   }

   if (if_debug() && ret != default_value)
       printf("Setting %s to %d\n", env_name, ret);

   return ret;
}

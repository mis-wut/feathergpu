#include "memory.cuh"

void mmCudaMallocHost(mmManager &manager, void **data, unsigned long size)
{
    gpuErrchk(cudaMallocHost(data, size));
    allocation_info el;
    el.data = *data;
    el.size = size;
    el.device = 0;
    el.freed = false;
    manager.push_front(el);
}
void mmCudaMalloc(mmManager &manager, void **data, unsigned long size)
{
    gpuErrchk(cudaMalloc(data, size));
    allocation_info el;
    el.data = *data;
    el.size = size;
    el.device = 1;
    el.freed = false;
    manager.push_front(el);
}

void __mmCudaFreeInternal(allocation_info *d)
{
    if (!(*d).freed) {
        if ((*d).device) {
            gpuErrchk(cudaFree((*d).data));
        } else {
            gpuErrchk(cudaFreeHost((*d).data));
        }
        (*d).freed = true;
    }
}

void mmCudaFreeAll(mmManager &manager)
{
   mmManager::iterator i;
   for(i=manager.begin(); i != manager.end(); ++i)
       __mmCudaFreeInternal(&(*i));
   manager.clear();
}

void mmCudaReportUsage( mmManager &manager)
{
    if (!if_debug()) return;

   mmManager::iterator i;
   unsigned long gpu_size = 0, cpu_size = 0;
   printf("Memory usage report\n");
   for(i=manager.begin(); i != manager.end(); ++i) {
       if (!(*i).freed) {
           if ((*i).device) {
               gpu_size += (*i).size;
               printf("GPU %ld\n", (*i).size);
           } else {
               cpu_size += (*i).size;
               printf("CPU %ld\n", (*i).size);
           }
       }
   }
   printf("Summary: GPU %ld, CPU %ld\n", gpu_size, cpu_size);
}

void mmCudaFree(mmManager &manager, void *ptr)
{
   mmManager::iterator i;
   for(i=manager.begin(); i != manager.end(); ++i)
       if ((*i).data == ptr) __mmCudaFreeInternal(&(*i));
}

void _cudaErrorCheck(const char *file, int line)
{
    gpuAssert( cudaPeekAtLastError(), file, line );
    gpuAssert( cudaDeviceSynchronize(), file, line );
}

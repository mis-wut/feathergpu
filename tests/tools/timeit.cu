#include "timeit.cuh"
#include <stdio.h>

void tiStart(tiManager &manager)
{
    timeit_info *el = (timeit_info *) malloc (sizeof(timeit_info));
    memset(el, 0 , sizeof(*el));

    cudaEventCreate( &(el->__start) );
    cudaEventCreate( &(el->__stop) );
    cudaEventRecord( el->__start, 0 );

    manager.push_front(el);
}

void tiEnd(tiManager &manager, const char * name)
{
    timeit_info *el = manager.front();

    cudaEventRecord( el->__stop, 0 );
    cudaEventSynchronize( el->__stop );
    cudaEventElapsedTime( &(el->__elapsedTime), el->__start, el->__stop );

    el->name = strdup(name);
}
void tiPreatyPrint(tiManager &manager)
{
   tiManager::iterator i;
   float sum = 0.0;
   for(i=manager.begin(); i != manager.end(); ++i)
       sum += (*i)->__elapsedTime;
   printf("Elapsed time: %f [ms] (", sum);
   for(i=manager.begin(); i != manager.end(); ++i)
       printf("%s %f, ", (*i)->name, (*i)->__elapsedTime );
   printf(")\n");
}

void tiPreatyPrintThrougput(tiManager &manager, long data_size)
{
   tiManager::iterator i;
   float sum = 0.0;
   int gb = 1024 * 1024 * 1024, sec=1000;

   // Print sum
   for(i=manager.begin(); i != manager.end(); ++i)
       sum += (*i)->__elapsedTime;
   printf("Time; %.2f; ms; Data size; %ld; bytes;", sum, data_size);

   // Print sum for operations marked as *
   sum = 0.0;
   for(i=manager.begin(); i != manager.end(); ++i)
       if ((*i)->name[0] == '*')
           sum += (*i)->__elapsedTime;
   printf("*; %.2f; ms; %.2f; GB/s; ", sum, ((float)data_size / gb)/ (sum/sec));

   // Print each operation
   for(i=manager.begin(); i != manager.end(); ++i)
       printf("%s; %.2f;ms; %.2f;GB/s; ", (*i)->name, (*i)->__elapsedTime, ((float)data_size / gb) / ((float)(*i)->__elapsedTime / sec) );
   printf("\n");
}

void tiClear(tiManager &manager)
{
   tiManager::iterator i;
   for(i=manager.begin(); i != manager.end(); ++i) {
       free(*i);
   }
   manager.clear();
}

/*TODO: rewrite timeit.cu to something like this */
#include <string>
#include <list>
class Event
{
public:
    Event (std::string name):
    name(name)
    {
        cudaEventCreate( &(_start) );
        cudaEventCreate( &(_stop) );
    };

    void start() {
        cudaEventRecord( _start, 0 );
    };

    void stop() {
        cudaEventRecord( _stop, 0 );
        cudaEventSynchronize( _stop );
        cudaEventElapsedTime( &(_elapsedTime), _start, _stop );
    }

    virtual ~Event ();

protected:
    float _elapsedTime;
    cudaEvent_t _start;
    cudaEvent_t _stop;
    std::string name;
};

class EventBlock
{
public:
    EventBlock (std::string name):
        name(name)
    {
        eventList = new std::list<Event *> ();

    };
    virtual ~EventBlock ();
    void startEvent(std::string name)
    {
        Event *v = new Event(name);
        v->start();
        this->eventList->push_front(v);
    };

    void stopEvent()
    {
        Event *v = this->eventList->front();
        v->stop();
    }

private:
    std::string name;
    std::list <Event *> *eventList;
};

class PerformanceLogger
{
public:
    PerformanceLogger (std::string name):
        name(name)
    {
        eventBlockList = new std::list<EventBlock *> ();
    };
    virtual ~PerformanceLogger ();

    void startBlock(std::string name)
    {
        EventBlock *v = new EventBlock(name);
        this->eventBlockList->push_front(v);
    };

    void startEvent(std::string name)
    {
        this->eventBlockList->front()->startEvent(name);
    }

    void stopEvent(std::string name)
    {
        this->eventBlockList->front()->stopEvent();
    }

private:
    std::string name;
    std::list <EventBlock *> *eventBlockList;
};


/*class PerformanceFileLogger*/
/*{*/
/*public:*/
    /*PerformanceLogger (std::string fname){*/
         /*logFile.open (fname);*/
    /*}*/

    /*~PerformanceLogger (){*/
        /*if (logFile.is_open()) {*/
            /*logFile << std::endl << std::endl;*/
            /*logFile.close();*/
        /*} */
    /*}*/

    /*friend FileLogger &operator << (FileLogger &logger, const char *text) {*/
        /*logger.logFile << text << std::endl;*/
        /*return logger;*/
    /*}*/

    /*// Make it Non Copyable (or you can inherit from sf::NonCopyable if you want)*/
    /*FileLogger (const FileLogger &) = delete;*/
    /*FileLogger &operator= (const FileLogger &) = delete;*/

/*private:*/
    /*std::ofstream logFile;*/
/*};*/

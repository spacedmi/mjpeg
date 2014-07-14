#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <ctype.h>

#define AVI 1

using namespace cv;
using namespace std;

class MjpegWriter
{
public:
    FILE * outFile;
    uchar outformat, outfps;
    MjpegWriter();
    int Open(const char* outfile, uchar format, uchar fps);
    int Close();
    int Write(const Mat Im);
    bool isOpened();
private:
    bool isOpen;
};
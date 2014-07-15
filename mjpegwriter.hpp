#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <intsafe.h>
#include <ctype.h>

#define AVI 1

using namespace cv;
using namespace std;

class MjpegWriter
{
public:
    FILE * outFile;
    uchar outformat, outfps;
    uint width, height;
    MjpegWriter();
    int Open(const char* outfile, uchar format, uchar fps);
    int Write(const Mat Im);
    int Close();
    bool isOpened();
private:
    bool isOpen;
};

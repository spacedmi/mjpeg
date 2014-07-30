#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "timer.hpp"
#include "mjpegwriter.hpp"

using namespace cv;
using namespace std;

#define TEST_MY 0

int main(int, char**)
{
	Rect rect(0, 0, 1920, 1080);
	Mat img(rect.size(), CV_8UC3);
    img = imread("1920x1080.jpg");
    img.size();
    int nframes = 100;
    jcodec::MjpegWriter * j = new jcodec::MjpegWriter();
    VideoWriter outputVideo;
    
    timer tt;
    tt.start();
    double ttotal = 0;
#if TEST_MY
    j->Open("out.avi", (uchar)10, img.size());
#else
    outputVideo.open("out2.avi", outputVideo.fourcc('M', 'J', 'P', 'G'), 10.0, img.size(), true);
#endif

	for (int i = 0; i < nframes; i++)
	{
        double tstart = (double)getTickCount();

#if TEST_MY
        j->Write(img);
#else
        outputVideo.write(img);
#endif
        double tend = (double)getTickCount();
        ttotal += tend - tstart;
        
        putchar('.');
        fflush(stdout);
	}

#if TEST_MY
    j->Close();
#else
    outputVideo.release();
#endif
    tt.stop();
    printf("time per frame (including file i/o)=%.1fms\n", (double)tt.get_elapsed_ms()/nframes);

	return 0;
}

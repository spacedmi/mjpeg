#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "timer.hpp"
#include "mjpegwriter.hpp"

using namespace cv;
using namespace std;

#define TEST_MY 1

int main(int, char**)
{
	Mat img = imread("1920x1080.jpg");
    Mat img_yuv, img_yuv444p;
    Rect rect(0, 0, img.cols, img.rows);
    int nframes = 10;
    jcodec::MjpegWriter * j = new jcodec::MjpegWriter();
    VideoWriter outputVideo;
    
    timer tt;
    tt.start();
    double ttotal = 0;

    cvtColor(img, img_yuv, COLOR_BGR2YUV);
    img_yuv444p.create(img.rows * 3 , img.cols, CV_8U);
    Mat planes[] =
    {
        img_yuv444p.rowRange(0, img.rows),
        img_yuv444p.rowRange(img.rows, img.rows * 2),
        img_yuv444p.rowRange(img.rows * 2, img.rows * 3)
    };
    split(img_yuv, planes);

#if TEST_MY
    j->Open("out.avi", (uchar)30, img_yuv.size(), jcodec::COLORSPACE_YUV444P);
#else
    outputVideo.open("out2.avi", outputVideo.fourcc('M', 'J', 'P', 'G'), 30.0, img.size(), true);
#endif

	for (int i = 0; i < nframes; i++)
	{
        double tstart = (double)getTickCount();

#if TEST_MY
        j->Write(img_yuv444p);
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

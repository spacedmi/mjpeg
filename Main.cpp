#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "jpge.hpp"
#include "jpge.hpp"
#include "jpgd.h"
#include "stb_image.c"
#include "timer.h"
#include <ctype.h>

using namespace cv;
using namespace std;

// Это для теста
static char s_log_filename[256];

static uint get_file_size(const char *pFilename)
{
    FILE *pFile = fopen(pFilename, "rb");
    if (!pFile) return 0;
    fseek(pFile, 0, SEEK_END);
    uint file_size = ftell(pFile);
    fclose(pFile);
    return file_size;
}

static void log_printf(const char *pMsg, ...)
{
    va_list args;

    va_start(args, pMsg);
    char buf[2048];
    vsnprintf(buf, sizeof(buf)-1, pMsg, args);
    buf[sizeof(buf)-1] = '\0';
    va_end(args);

    printf("%s", buf);

    if (s_log_filename[0])
    {
        FILE *pFile = fopen(s_log_filename, "a+");
        if (pFile)
        {
            fprintf(pFile, "%s", buf);
            fclose(pFile);
        }
    }
}

static void encode()
{
    // Тест jpge ------------------------------------------------------------------------------------------------------
    // Parse command line.
    bool run_exhausive_test = false;
    bool test_memory_compression = false;
    bool optimize_huffman_tables = false;
    int subsampling = -1;
    char output_filename[256] = "";
    bool use_jpgd = true;
    bool test_jpgd_decompression = false;

    int quality_factor = 50;
    const char* pSrc_filename = "test.png";
    const char* pDst_filename = "comp.jpg";

    // Load the source image.
    const int req_comps = 3; // request RGB image
    int width = 0, height = 0, actual_comps = 0;
    uint8 *pImage_data = stbi_load(pSrc_filename, &width, &height, &actual_comps, req_comps);
    if (!pImage_data)
    {
        log_printf("Failed loading file \"%s\"!\n", pSrc_filename);
        //return 0;
    }

    log_printf("Source file: \"%s\", image resolution: %ix%i, actual comps: %i\n", pSrc_filename, width, height, actual_comps);

    // Fill in the compression parameter structure.
    jpge::params params;
    params.m_quality = quality_factor;
    params.m_subsampling = (subsampling < 0) ? ((actual_comps == 1) ? jpge::Y_ONLY : jpge::H2V2) : static_cast<jpge::subsampling_t>(subsampling);
    params.m_two_pass_flag = optimize_huffman_tables;

    log_printf("Writing JPEG image to file: %s\n", pDst_filename);

    timer tm;

    // Now create the JPEG file.
    if (test_memory_compression)
    {
        int buf_size = width * height * 3; // allocate a buffer that's hopefully big enough (this is way overkill for jpeg)
        if (buf_size < 1024) buf_size = 1024;
        void *pBuf = malloc(buf_size);

        tm.start();
        if (!jpge::compress_image_to_jpeg_file_in_memory(pBuf, buf_size, width, height, req_comps, pImage_data, params))
        {
            log_printf("Failed creating JPEG data!\n");
            //return EXIT_FAILURE;
        }
        tm.stop();

        FILE *pFile = fopen(pDst_filename, "wb");
        if (!pFile)
        {
            log_printf("Failed creating file \"%s\"!\n", pDst_filename);
            //return EXIT_FAILURE;
        }

        if (fwrite(pBuf, buf_size, 1, pFile) != 1)
        {
            log_printf("Failed writing to output file!\n");
            //return EXIT_FAILURE;
        }

        if (fclose(pFile) == EOF)
        {
            log_printf("Failed writing to output file!\n");
            //return EXIT_FAILURE;
        }
    }
    else
    {
        tm.start();

        if (!jpge::compress_image_to_jpeg_file(pDst_filename, width, height, req_comps, pImage_data, params))
        {
            log_printf("Failed writing to output file!\n");
            //return EXIT_FAILURE;
        }
        tm.stop();
    }

    double total_comp_time = tm.get_elapsed_ms();

    const uint comp_file_size = get_file_size(pDst_filename);
    const uint total_pixels = width * height;
    log_printf("Compressed file size: %u, bits/pixel: %3.3f\n", comp_file_size, (comp_file_size * 8.0f) / total_pixels);

    // Конец теста jpge ------------------------------------------------------------------------------------------------------------------------------------
}
// Конец
static void help()
{
	cout << "\nThis program demostrates iterative construction of\n"
		"delaunay triangulation and voronoi tesselation.\n"
		"It draws a random set of points in an image and then delaunay triangulates them.\n"
		"Usage: \n"
		"./delaunay \n"
		"\nThis program builds the traingulation interactively, you may stop this process by\n"
		"hitting any key.\n";
}

static void draw_subdiv_point(Mat& img, Point2f fp, Scalar color)
{
	circle(img, fp, 3, color, FILLED, LINE_8, 0);
}

static void draw_subdiv(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{
#if 1
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);

	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		line(img, pt[0], pt[1], delaunay_color, 1, LINE_AA, 0);
		line(img, pt[1], pt[2], delaunay_color, 1, LINE_AA, 0);
		line(img, pt[2], pt[0], delaunay_color, 1, LINE_AA, 0);
	}
#else
	vector<Vec4f> edgeList;
	subdiv.getEdgeList(edgeList);
	for (size_t i = 0; i < edgeList.size(); i++)
	{
		Vec4f e = edgeList[i];
		Point pt0 = Point(cvRound(e[0]), cvRound(e[1]));
		Point pt1 = Point(cvRound(e[2]), cvRound(e[3]));
		line(img, pt0, pt1, delaunay_color, 1, LINE_AA, 0);
	}
#endif
}

static void locate_point(Mat& img, Subdiv2D& subdiv, Point2f fp, Scalar active_color)
{
	int e0 = 0, vertex = 0;

	subdiv.locate(fp, e0, vertex);

	if (e0 > 0)
	{
		int e = e0;
		do
		{
			Point2f org, dst;
			if (subdiv.edgeOrg(e, &org) > 0 && subdiv.edgeDst(e, &dst) > 0)
				line(img, org, dst, active_color, 3, LINE_AA, 0);

			e = subdiv.getEdge(e, Subdiv2D::NEXT_AROUND_LEFT);
		} while (e != e0);
	}

	draw_subdiv_point(img, fp, active_color);
}


static void paint_voronoi(Mat& img, Subdiv2D& subdiv)
{
	vector<vector<Point2f> > facets;
	vector<Point2f> centers;
	subdiv.getVoronoiFacetList(vector<int>(), facets, centers);

	vector<Point> ifacet;
	vector<vector<Point> > ifacets(1);

	for (size_t i = 0; i < facets.size(); i++)
	{
		ifacet.resize(facets[i].size());
		for (size_t j = 0; j < facets[i].size(); j++)
			ifacet[j] = facets[i][j];

		Scalar color;
		color[0] = rand() & 255;
		color[1] = rand() & 255;
		color[2] = rand() & 255;
		fillConvexPoly(img, ifacet, color, 8, 0);

		ifacets[0] = ifacet;
		polylines(img, ifacets, true, Scalar(), 1, LINE_AA, 0);
		circle(img, centers[i], 3, Scalar(), FILLED, LINE_AA, 0);
	}
}


int main(int, char**)
{
	help();

	Scalar active_facet_color(0, 0, 255), delaunay_color(255, 255, 255);
	Rect rect(0, 0, 600, 600);

	Subdiv2D subdiv(rect);
	Mat img(rect.size(), CV_8UC3);

	img = Scalar::all(0);
	string win = "Delaunay Demo";
	string win2 = "Delaunay Dec Demo";
	//imshow(win, img);

	for (int i = 0; i < 200; i++)
	{
		Point2f fp((float)(rand() % (rect.width - 10) + 5),
			(float)(rand() % (rect.height - 10) + 5));

		locate_point(img, subdiv, fp, active_facet_color);
		//imshow(win, img);

		if (waitKey(100) >= 0)
			break;

		subdiv.insert(fp);

		img = Scalar::all(0);
		draw_subdiv(img, subdiv, delaunay_color);
		imshow(win, img);

		if (waitKey(100) >= 0)
			break;
	}

	img = Scalar::all(0);
	paint_voronoi(img, subdiv);
	//imshow(win, img);

	waitKey(0);

	return 0;
}

#include "jcodec.hpp"

using namespace cv;
using namespace std;

Jcodec::Jcodec(int qual = 2)
{
	quality = qual;
}

Mat Jcodec::encJPEG(Mat InputIm, int quality)
{
	// Тут должна быть проверка на количество каналов и число бит на канал ююю

	cvtColor(InputIm, InputIm, COLOR_BGR2YCrCb);
	InputIm = decimation(InputIm);
	cvtColor(InputIm, InputIm, COLOR_YCrCb2BGR);
	Mat DCTmat = genDCT();
	Mat Qmat = genQuant(quality);
	return InputIm;
}

Mat Jcodec::decimation(Mat InputIm)		// Прореживание блоками 2х2
{
	if ((InputIm.rows < 2) || (InputIm.cols < 2)) return InputIm;
	Mat DecIm = InputIm.clone();
	uchar Cr, Cb;
	int x, y;
	for (y = 0; y < DecIm.rows; y += 2)
	{
		uchar* row1 = DecIm.ptr(y);
		uchar* row2 = DecIm.ptr(y);
		for (x = 0; x < DecIm.cols; x += 2)
		{
			/*Cr = (DecIm.at<Vec3b>(y, x)[1] + DecIm.at<Vec3b>(y, x + 1)[1] + DecIm.at<Vec3b>(y + 1, x)[1] + DecIm.at<Vec3b>(y + 1, x + 1)[1]) >> 2;
			Cb = (DecIm.at<Vec3b>(y, x)[2] + DecIm.at<Vec3b>(y, x + 1)[2] + DecIm.at<Vec3b>(y + 1, x)[2] + DecIm.at<Vec3b>(y + 1, x + 1)[2]) >> 2;*/
			Cr = (row1[x * 3 + 1] + row1[(x + 1) * 3 + 1] + row2[x * 3 + 1] + row2[(x + 1) * 3 + 1]) >> 2;
			Cb = (row1[x * 3 + 2] + row1[(x + 1) * 3 + 2] + row2[x * 3 + 2] + row2[(x + 1) * 3 + 2]) >> 2;
			DecIm.at<Vec3b>(y, x)[1] = DecIm.at<Vec3b>(y, x + 1)[1] = DecIm.at<Vec3b>(y + 1, x)[1] = DecIm.at<Vec3b>(y + 1, x + 1)[1] = Cr;
			DecIm.at<Vec3b>(y, x)[2] = DecIm.at<Vec3b>(y, x + 1)[2] = DecIm.at<Vec3b>(y + 1, x)[2] = DecIm.at<Vec3b>(y + 1, x + 1)[2] = Cb;
		}
	}
	if (DecIm.cols % 2)
		for (y = 0; y < DecIm.rows; y += 2)
		{
			Cr = (DecIm.at<Vec3b>(y, DecIm.cols)[1] + DecIm.at<Vec3b>(y + 1, DecIm.cols)[1]) >> 1;
			Cb = (DecIm.at<Vec3b>(y, DecIm.cols)[2] + DecIm.at<Vec3b>(y + 1, DecIm.cols)[2]) >> 1;
			DecIm.at<Vec3b>(y, DecIm.cols)[1] = DecIm.at<Vec3b>(y + 1, DecIm.cols)[1] = Cr;
			DecIm.at<Vec3b>(y, DecIm.cols)[2] = DecIm.at<Vec3b>(y + 1, DecIm.cols)[2] = Cb;
		}
	if (DecIm.rows % 2)
		for (x = 0; x < DecIm.cols; x += 2)
		{
			Cr = (DecIm.at<Vec3b>(DecIm.rows, x)[1] + DecIm.at<Vec3b>(DecIm.rows, x + 1)[1]) >> 1;
			Cb = (DecIm.at<Vec3b>(DecIm.rows, x)[2] + DecIm.at<Vec3b>(DecIm.rows, x + 1)[2]) >> 1;
			DecIm.at<Vec3b>(DecIm.rows, x)[1] = DecIm.at<Vec3b>(DecIm.rows, x + 1)[1] = Cr;
			DecIm.at<Vec3b>(DecIm.rows, x)[2] = DecIm.at<Vec3b>(DecIm.rows, x + 1)[2] = Cb;
		}
	return DecIm;
}

Mat Jcodec::genDCT()		// Генерация матрицы ДКП (8х8)
{
	int row, col;
	Mat DCT(Size(8, 8), CV_32FC1);
	for (col = 0; col < 8; col++)
		DCT.at<float>(0, col) = 1 / sqrt(8.0f);
	for (row = 1; row < 8; row++)
		for (col = 0; col < 8; col++)
			DCT.at<float>(row, col) = sqrt(0.25f) * cos((2.0f * col + 1.0f) * row * CV_PI / 16.0f);
	return DCT;
}

Mat Jcodec::genQuant(int quality)		// Генерация матрицы квантования, где q - коэффициент качества (8х8)
{
	Mat Quant(Size(8, 8), CV_32SC1);
	for (int row = 0; row < 8; row++)
	for (int col = 0; col < 8; col++)
		Quant.at<int>(row, col) = 1 + ((1 + row + col) * quality);
	return Quant;
}

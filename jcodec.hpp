#pragma once
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>

using namespace cv;
class Jcodec
{
public:
	Jcodec(int quality);

	int quality;

	static Mat encJPEG(Mat InputIm, int qual);				// Общий метод кодирования
	static void decJPEG(FILE * Jfile);						// Общий метод декодирования
protected:
	static Mat decimation(Mat InputIm);						// Прореживание 4:2:0
	static Mat genDCT();									// генерация матрицы ДКП
	static Mat genQuant(int quality = 2);					// Генерация матрицы квантования
};
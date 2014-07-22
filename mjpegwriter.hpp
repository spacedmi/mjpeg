#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <intsafe.h>
#include <ctype.h>
#include <vector>


#define AVI 1

using namespace cv;
using namespace std;

class MjpegWriter
{
public:
    MjpegWriter();
    int Open(char* outfile, uchar format, uchar fps);
    int Write(const Mat &Im);
    int Close();
    bool isOpened();
private:
    FILE * outFile;
    char* outfileName;
    int outformat, outfps;
    int width, height, type, FrameNum;
    int chunkPointer, moviPointer;
    vector<int> FrameOffset, FrameSize;
    // Переменные позиций указателей данных о размерах chunk'ов
    int FrameNumIndex, FrameNumDwLengthIndex;
    int AVIChunkSizeIndex[10];
    int curChunkNum;

    bool isOpen, isRecStarted;

    int toJPGframe(const uchar * data, uint width, uint height, int step, void *& pBuf);
    void StartWriteAVI();
    void WriteStreamHeader();
    void WriteIndex();
    void WriteODMLIndex();
    void FinishWriteAVI();
    void PutInt(int elem);
    void PutShort(short elem);
    void StartWriteChunk(int fourcc);
    void EndWriteChunk();
};

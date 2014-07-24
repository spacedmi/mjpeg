#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <intsafe.h>
#include <ctype.h>
#include <vector>

using namespace cv;
using namespace std;

namespace jcodec
{

    class MjpegWriter
    {
    public:
        MjpegWriter();
        int Open(char* outfile, uchar format, uchar fps);
        int Write(const Mat &Im);
        int Close();
        bool isOpened();
    private:
        double tencoding;
        FILE * outFile;
        char* outfileName;
        int outformat, outfps, quality;
        int width, height, type, FrameNum;
        int chunkPointer, moviPointer;
        vector<int> FrameOffset, FrameSize;
        int FrameNumIndex, FrameNumDwLengthIndex; // Frame number positions
        int AVIChunkSizeIndex[10];                // Chunk sizes positions
        int curChunkNum;                          // Current number opened chunks

        bool isOpen, isRecStarted;

        int toJPGframe(const uchar * data, uint width, uint height, int step, void *& pBuf);
        void StartWriteAVI();
        void WriteStreamHeader();
        void WriteIndex();
        bool WriteFrame(const Mat & Im);
        void WriteODMLIndex();
        void FinishWriteAVI();
        void PutInt(int elem);
        void PutShort(short elem);
        void StartWriteChunk(int fourcc);
        void EndWriteChunk();
    };
}
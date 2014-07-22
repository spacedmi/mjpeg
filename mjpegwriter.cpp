#include <fstream>
#include "mjpegwriter.hpp"
#include "jpge.hpp"

#define fourCC(a,b,c,d) ( (int) ((uchar(d)<<24) | (uchar(c)<<16) | (uchar(b)<<8) | uchar(a)) )
#define DIM(arr) (sizeof(arr)/sizeof(arr[0]))

static const float NUM_MICROSEC_PER_SEC = 1000000.0f;
static const int STREAMS = 1;
static const int AVIH_STRH_SIZE = 56;
static const int STRF_SIZE = 40;
static const int AVI_DWFLAG = 16;
static const int AVI_DWSCALE = 1;
static const int AVI_DWQUALITY = -1;
static const int AVI_BITS_PER_PIXEL = 24;
static const int AVI_BIPLANES = 1;
static const int AVI_BICLRUSED = 256;
static const int JUNK_SEEK = 4096;

#define MJPG_COMPRESSION 0x47504a4d
#define AVIIF_KEYFRAME 0x10

MjpegWriter::MjpegWriter() : isOpen(false), outFile(0), outformat(AVI), outfps(20), curChunkNum(0),
             isRecStarted(false), FrameNum(0) { }

int MjpegWriter::Open(char* outfile, uchar format, uchar fps)
{
    if (isOpen) return -4;
    if (fps < 1) return -3;
    switch (format)
    {
        case AVI:
        {
            if (!(outFile = fopen(outfile, "wb+")))
                return -1;
            outfps = fps;
            break;
        }
        default: 
            return -2;
    }
    
    isOpen = true;
    isRecStarted = false;
    outfileName = outfile;
    return 1;
}

int MjpegWriter::Close()
{
    if (FrameNum == 0)
    {
        remove(outfileName);
        isOpen = false;
        if (fclose(outFile))
            return -1;
        return 1;
    }
    EndWriteChunk(); // end LIST 'movi'
    WriteIndex();
    FinishWriteAVI();
    if (fclose(outFile))
        return -1;
    isOpen = false;
    isRecStarted = false;
    return 1;
}

int MjpegWriter::Write(const Mat & Im)
{
    if (!isOpen) return -1;
    if (!isRecStarted) // If first frame
    {
        width = Im.cols;
        height = Im.rows;
        //type = Im.type;
        isRecStarted = true;
        // for AVI:
        StartWriteAVI();
        WriteStreamHeader();
    }
    chunkPointer = ftell(outFile);
    StartWriteChunk(fourCC('0', '0', 'd', 'c'));

    // Frame data
    void *pBuf = 0;
    int pBufSize;
    if ((pBufSize = toJPGframe(Im.data, width, height, 12, pBuf)) < 0)
        return -2;
    fwrite(pBuf, pBufSize, 1, outFile);

    FrameOffset.push_back(chunkPointer - moviPointer);
    FrameSize.push_back(ftell(outFile) - chunkPointer - 8);       // Size excludes '00dc' and size field
    FrameNum++;
    EndWriteChunk(); // end '00dc'
    return 1;
}

bool MjpegWriter::isOpened()
{
    return isOpen;
}

void MjpegWriter::StartWriteAVI()
{
    StartWriteChunk(fourCC('R', 'I', 'F', 'F'));
    PutInt(fourCC('A', 'V', 'I', ' '));
    StartWriteChunk(fourCC('L', 'I', 'S', 'T'));
    PutInt(fourCC('h', 'd', 'r', 'l'));
    PutInt(fourCC('a', 'v', 'i', 'h'));
    PutInt(AVIH_STRH_SIZE);
    PutInt((int)(NUM_MICROSEC_PER_SEC / outfps));
    PutInt(0);
    PutInt(0);
    PutInt(AVI_DWFLAG);
    PutInt(0);
    PutInt(0);
    PutInt(STREAMS);
    PutInt(0);
    PutInt(width);
    PutInt(height);
    PutInt(0);
    PutInt(0);
    PutInt(0);
    PutInt(0);
}

void MjpegWriter::WriteStreamHeader()
{
    // strh
    StartWriteChunk(fourCC('L', 'I', 'S', 'T'));
    PutInt(fourCC('s', 't', 'r', 'l'));
    PutInt(fourCC('s', 't', 'r', 'h'));
    PutInt(AVIH_STRH_SIZE);
    PutInt(fourCC('v', 'i', 'd', 's'));
    PutInt(fourCC('M', 'J', 'P', 'G'));
    PutInt(0);
    PutInt(0);
    PutInt(0);
    PutInt(AVI_DWSCALE);
    PutInt(outfps);
    PutInt(0);
    FrameNumDwLengthIndex = ftell(outFile);
    PutInt(0);
    PutInt(0);
    PutInt(AVI_DWQUALITY);
    PutInt(0);
    PutShort(0);
    PutShort(0);
    PutShort(0);
    PutShort(0);
    // strf (use the BITMAPINFOHEADER for video)
    StartWriteChunk(fourCC('s', 't', 'r', 'f'));
    PutInt(STRF_SIZE);
    PutInt(width);
    PutInt(height);
    PutShort(AVI_BIPLANES);
    PutShort(AVI_BITS_PER_PIXEL);
    PutInt((int)MJPG_COMPRESSION);
    PutInt(width * height * 3);
    PutInt(0);
    PutInt(0);
    PutInt(AVI_BICLRUSED);
    PutInt(0);
    EndWriteChunk(); // end strf
    EndWriteChunk(); // end strl
    EndWriteChunk(); // end hdrl
    // JUNK
    StartWriteChunk(fourCC('J', 'U', 'N', 'K'));
    fseek(outFile, JUNK_SEEK, SEEK_SET);
    EndWriteChunk(); // end JUNK
    // movi
    StartWriteChunk(fourCC('L', 'I', 'S', 'T'));
    moviPointer = ftell(outFile);
    PutInt(fourCC('m', 'o', 'v', 'i'));
}

void MjpegWriter::WriteIndex()
{
    // old style AVI index. Must be Open-DML index
    StartWriteChunk(fourCC('i', 'd', 'x', '1'));
    for (int i = 0; i < FrameNum; i++)
    {
        PutInt(fourCC('0', '0', 'd', 'c'));
        PutInt((int)AVIIF_KEYFRAME);
        PutInt(FrameOffset[i]);
        PutInt(FrameSize[i]);
    }
    EndWriteChunk(); // End idx1
}

void MjpegWriter::WriteODMLIndex()
{
    StartWriteChunk(fourCC('i', 'n', 'd', 'x'));
    for (int i = 0; i < FrameNum; i++)
    {
       
    }
    EndWriteChunk(); // End indx
}

void MjpegWriter::FinishWriteAVI()
{
    EndWriteChunk(); // end RIFF
    // Record frames number to AVI Header
    fseek(outFile, FrameNumIndex, SEEK_SET);
    PutInt(FrameNum);
    // Record frames number to Stream Header
    fseek(outFile, FrameNumDwLengthIndex, SEEK_SET);
    PutInt(FrameNum);
}

void MjpegWriter::PutInt(int elem)
{
    fwrite(&elem, 1, sizeof(elem), outFile);
}

void MjpegWriter::PutShort(short elem)
{
    fwrite(&elem, 1, sizeof(elem), outFile);
}

void MjpegWriter::StartWriteChunk(int fourcc)
{
    long fpos;
    if (fourcc != 0)
    {
        fpos = ftell(outFile) + 4;
        PutInt(fourcc);
        PutInt(0);
    }
    else
    {
        fpos = ftell(outFile);
        PutInt(0);
    }
    AVIChunkSizeIndex[curChunkNum++] = (uint)fpos;
}

void MjpegWriter::EndWriteChunk()
{
    curChunkNum--;
    long curPointer = ftell(outFile);
    fseek(outFile, AVIChunkSizeIndex[curChunkNum], SEEK_SET);
    int size = (int)(curPointer - (AVIChunkSizeIndex[curChunkNum] + 4));
    PutInt(size);
    fseek(outFile, curPointer, SEEK_SET);
}

int MjpegWriter::toJPGframe(const uchar * data, uint width, uint height, int step, void *& pBuf)
{
    int quality_factor = 70;
    int subsampling = -1;
    int actual_comps = 0;
    const int req_comps = 3; // request RGB image
    bool optimize_huffman_tables = false;
    jpge::params param;
    param.m_quality = quality_factor;
    param.m_subsampling = (subsampling < 0) ? ((actual_comps == 1) ? jpge::Y_ONLY : jpge::H2V2) : static_cast<jpge::subsampling_t>(subsampling);
    param.m_two_pass_flag = optimize_huffman_tables;
    int buf_size = width * height * 3; // allocate a buffer that's hopefully big enough (this is way overkill for jpeg)
    if (buf_size < 1024) buf_size = 1024;
    pBuf = malloc(buf_size);
    if (!jpge::compress_image_to_jpeg_file_in_memory(pBuf, buf_size, width, height, req_comps, data, param))
        return -1;
    return buf_size;
}
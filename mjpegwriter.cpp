
#include "mjpegwriter.hpp"
#include "opencv2/core/utility.hpp"
#include <smmintrin.h>

namespace jcodec
{

#define SSE 1
#define fourCC(a,b,c,d) ( (int) ((uchar(d)<<24) | (uchar(c)<<16) | (uchar(b)<<8) | uchar(a)) )
#define DIM(arr) (sizeof(arr)/sizeof(arr[0]))
#define BSWAP(v)    (((v)<<24)|(((v)&0xff00)<<8)| \
    (((v) >> 8) & 0xff00) | ((unsigned)(v) >> 24))
#define DCT_DESCALE(x, n) (((x) + (((int)1) << ((n) - 1))) >> (n)) 
#define fix(x, n)   (int)((x)*(1 << (n)) + .5);
#define fix1(x, n)  (x)
#define fixmul(x)   (x)

    static const float NUM_MICROSEC_PER_SEC = 1000000.0f;
    static const int STREAMS = 1;
    static const int AVIH_STRH_SIZE = 56;
    static const int STRF_SIZE = 40;
    static const int AVI_DWFLAG = 0x00000910;
    static const int AVI_DWSCALE = 1;
    static const int AVI_DWQUALITY = -1;
    static const int AVI_BITS_PER_PIXEL = 24;
    static const int AVI_BIPLANES = 1;
    static const int JUNK_SEEK = 4096;
    static const int AVIIF_KEYFRAME = 0x10;
    static const int MAX_BYTES_PER_SEC = 15552000;
    static const int SUG_BUFFER_SIZE = 1048576;
    static const int RBS_THROW_EOS = -123;  /* <end of stream> exception code */
    static const int RBS_THROW_FORB = -124;  /* <forrbidden huffman code> exception code */
    static const int RBS_HUFF_FORB = 2047;  /* forrbidden huffman code "value" */
    static const int fixb = 14;

    static const int C0_707 = fix(0.707106781f, fixb);
    static const int C0_924 = fix(0.923879533f, fixb);
    static const int C0_541 = fix(0.541196100f, fixb);
    static const int C0_382 = fix(0.382683432f, fixb);
    static const int C1_306 = fix(1.306562965f, fixb);

    static const int C1_082 = fix(1.082392200f, fixb);
    static const int C1_414 = fix(1.414213562f, fixb);
    static const int C1_847 = fix(1.847759065f, fixb);
    static const int C2_613 = fix(2.613125930f, fixb);

    static const int fixc = 12;
    static const int b_cb = fix(1.772, fixc);
    static const int g_cb = -fix(0.34414, fixc);
    static const int g_cr = -fix(0.71414, fixc);
    static const int r_cr = fix(1.402, fixc);

    static const int y_r = fix(0.299, fixc);
    static const int y_g = fix(0.587, fixc);
    static const int y_b = fix(0.114, fixc);

    static const int cb_r = -fix(0.1687, fixc);
    static const int cb_g = -fix(0.3313, fixc);
    static const int cb_b = fix(0.5, fixc);

    static const int cr_r = fix(0.5, fixc);
    static const int cr_g = -fix(0.4187, fixc);
    static const int cr_b = -fix(0.0813, fixc);

    static const int huff_val_shift = 20, huff_code_mask = (1 << huff_val_shift) - 1;
    static const int postshift = 14;

    static uchar clamp_table[1024];
    static bool init_clamp_table = false;

    static inline uchar clamp(int i) { if (static_cast<uint>(i) > 255U) { i = clamp_table[(i)+256]; } return static_cast<uchar>(i); }

    static const int BITS = 10, SCALE = 1 << BITS;
    static const float MAX_M = (float)(1 << (15 - BITS));
    static const short m00 = static_cast<short>(-0.0813f * SCALE), m01 = static_cast<short>(-0.4187f * SCALE),
        m02 = static_cast<short>(0.5f * SCALE), m10 = static_cast<short>(0.5f * SCALE),
        m11 = static_cast<short>(-0.3313f * SCALE), m12 = static_cast<short>(-0.1687f * SCALE),
        m20 = static_cast<short>(0.114f * SCALE), m21 = static_cast<short>(0.587f  * SCALE),
        m22 = static_cast<short>(0.299f * SCALE), Hm03 = static_cast<int>(0.5f * SCALE),
        Hm13 = static_cast<int>(0.5f * SCALE), Hm23 = static_cast<int>(0.5f * SCALE),
        Hm = static_cast<int>(0.5f * SCALE);

    //  Standard JPEG quantization tables
    static const uchar jpegTableK1_T[] =
    {
        16, 12, 14, 14, 18, 24, 49, 72,
        11, 12, 13, 17, 22, 35, 64, 92,
        10, 14, 16, 22, 37, 55, 78, 95,
        16, 19, 24, 29, 56, 64, 87, 98,
        24, 26, 40, 51, 68, 81, 103, 112,
        40, 58, 57, 87, 109, 104, 121, 100,
        51, 60, 69, 80, 103, 113, 120, 103,
        61, 55, 56, 62, 77, 92, 101, 99
    };

    static const uchar jpegTableK2_T[] =
    {
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99
    };

    // Standard Huffman tables

    // ... for luma DCs.
    static const uchar jpegTableK3[] =
    {
        0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    };

    // ... for chroma DCs.
    static const uchar jpegTableK4[] =
    {
        0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    };

    // ... for luma ACs.
    static const uchar jpegTableK5[] =
    {
        0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
        0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
        0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
        0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
        0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
        0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
        0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
        0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
        0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
        0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
        0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa
    };

    // ... for chroma ACs  
    static const uchar jpegTableK6[] =
    {
        0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119,
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
        0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
        0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
        0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
        0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
        0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
        0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
        0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
        0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
        0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
        0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
        0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
        0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
        0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
        0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa
    };

    static const uchar zigzag[] =
    {
        0, 8, 1, 2, 9, 16, 24, 17, 10, 3, 4, 11, 18, 25, 32, 40,
        33, 26, 19, 12, 5, 6, 13, 20, 27, 34, 41, 48, 56, 49, 42, 35,
        28, 21, 14, 7, 15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30,
        23, 31, 38, 45, 52, 59, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63,
        63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63
    };

    static const uchar SSE_zigzag[] =
    {
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
        63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63
    };
    static const int idct_prescale[] =
    {
        16384, 22725, 21407, 19266, 16384, 12873, 8867, 4520,
        22725, 31521, 29692, 26722, 22725, 17855, 12299, 6270,
        21407, 29692, 27969, 25172, 21407, 16819, 11585, 5906,
        19266, 26722, 25172, 22654, 19266, 15137, 10426, 5315,
        16384, 22725, 21407, 19266, 16384, 12873, 8867, 4520,
        12873, 17855, 16819, 15137, 12873, 10114, 6967, 3552,
        8867, 12299, 11585, 10426, 8867, 6967, 4799, 2446,
        4520, 6270, 5906, 5315, 4520, 3552, 2446, 1247
    };

    static const char jpegHeader[] =
        "\xFF\xD8"  // SOI  - start of image
        "\xFF\xE0"  // APP0 - jfif extention
        "\x00\x10"  // 2 bytes: length of APP0 segment
        "JFIF\x00"  // JFIF signature
        "\x01\x02"  // version of JFIF
        "\x00"      // units = pixels ( 1 - inch, 2 - cm )
        "\x00\x01\x00\x01" // 2 2-bytes values: x density & y density
        "\x00\x00"; // width & height of thumbnail: ( 0x0 means no thumbnail)

    /////////////////////// MjpegWriter ///////////////////

    MjpegWriter::MjpegWriter() : isOpen(false), outFile(0), outformat(1), outfps(20),
        FrameNum(0), quality(80), NumOfChunks(10), InputColorSpace(COLORSPACE_BGR){}

    int MjpegWriter::Open(char* outfile, uchar fps, Size ImSize, colorspace ColorSpace)
    {
        tencoding = 0;
        if (isOpen) return -4;
        if (fps < 1) return -3;
        if (!(outFile = fopen(outfile, "wb+")))
            return -1;
        outfps = fps;
        width = ImSize.width;
        height = ImSize.height;
        InputColorSpace = ColorSpace;

        StartWriteAVI();
        WriteStreamHeader();

        isOpen = true;
        outfileName = outfile;
        return 1;
    }

    int MjpegWriter::Close()
    {
        if (outFile == 0) return -1;
        if (FrameNum == 0)
        {
            remove(outfileName);
            isOpen = false;
            if (fclose(outFile))
                return -2;
            return 1;
        }
        printf("encoding time per frame = %.1fms\n", tencoding * 1000 / FrameNum / getTickFrequency());
        EndWriteChunk(); // end LIST 'movi'
        WriteIndex();
        FinishWriteAVI();
        if (fclose(outFile))
            return -1;
        isOpen = false;
        FrameNum = 0;
        return 1;
    }

    int MjpegWriter::Write(const Mat & Im)
    {
        if (!isOpen) return -1;
        if(!WriteFrame(Im))
            return -2;
        return 1;
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
        PutInt(MAX_BYTES_PER_SEC);
        PutInt(0);
        PutInt(AVI_DWFLAG);
        FrameNumIndexes.push_back(ftell(outFile));
        PutInt(0);
        PutInt(0);
        PutInt(STREAMS);
        PutInt(SUG_BUFFER_SIZE);
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
        FrameNumIndexes.push_back(ftell(outFile));
        PutInt(0);
        PutInt(SUG_BUFFER_SIZE);
        PutInt(AVI_DWQUALITY);
        PutInt(0);
        PutShort(0);
        PutShort(0);
        PutShort(width);
        PutShort(height);
        // strf (use the BITMAPINFOHEADER for video)
        StartWriteChunk(fourCC('s', 't', 'r', 'f'));
        PutInt(STRF_SIZE);
        PutInt(width);
        PutInt(height);
        PutShort(AVI_BIPLANES);
        PutShort(AVI_BITS_PER_PIXEL);
        PutInt(fourCC('M', 'J', 'P', 'G'));
        PutInt(width * height * 3);
        PutInt(0);
        PutInt(0);
        PutInt(0);
        PutInt(0);
        // Must be indx chunk
        EndWriteChunk(); // end strf
        EndWriteChunk(); // end strl

        // odml
        StartWriteChunk(fourCC('L', 'I', 'S', 'T'));
        PutInt(fourCC('o', 'd', 'm', 'l'));
        StartWriteChunk(fourCC('d', 'm', 'l', 'h'));
        FrameNumIndexes.push_back(ftell(outFile));
        PutInt(0);
        PutInt(0);
        EndWriteChunk(); // end dmlh
        EndWriteChunk(); // end odml

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
    
    bool MjpegWriter::WriteFrame(const Mat & Im)
    {
        chunkPointer = ftell(outFile);
        StartWriteChunk(fourCC('0', '0', 'd', 'c'));
        // Frame data
        void *pBuf = 0;
        int pBufSize;
        double t = (double)getTickCount();
        if ((pBufSize = toJPGframe(Im.data, width, height, pBuf)) < 0)
            return false;
        tencoding += (double)getTickCount() - t;
        fwrite(pBuf, pBufSize, 1, outFile);
        FrameOffset.push_back(chunkPointer - moviPointer);
        FrameSize.push_back(ftell(outFile) - chunkPointer - 8);       // Size excludes '00dc' and size field
        FrameNum++;
        EndWriteChunk(); // end '00dc'
        free(pBuf);
        return true;
    }

    void MjpegWriter::WriteIndex()
    {
        // old style AVI index. Must be Open-DML index
        StartWriteChunk(fourCC('i', 'd', 'x', '1'));
        for (int i = 0; i < FrameNum; i++)
        {
            PutInt(fourCC('0', '0', 'd', 'c'));
            PutInt(AVIIF_KEYFRAME);
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
        // Record frames numbers to AVI Header
        long curPointer = ftell(outFile);
        while (!FrameNumIndexes.empty())
        {
            fseek(outFile, FrameNumIndexes.back(), SEEK_SET);
            PutInt(FrameNum);
            FrameNumIndexes.pop_back();
        }
        fseek(outFile, curPointer, SEEK_SET);
        EndWriteChunk(); // end RIFF
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
            PutInt(fourcc);
            fpos = ftell(outFile);
            PutInt(0);
        }
        else
        {
            fpos = ftell(outFile);
            PutInt(0);
        }
        AVIChunkSizeIndex.push_back((int)fpos);
    }

    void MjpegWriter::EndWriteChunk()
    {
        if (!AVIChunkSizeIndex.empty())
        {
            long curPointer = ftell(outFile);
            fseek(outFile, AVIChunkSizeIndex.back(), SEEK_SET);
            int size = (int)(curPointer - (AVIChunkSizeIndex.back() + 4));
            PutInt(size);
            fseek(outFile, curPointer, SEEK_SET);
            AVIChunkSizeIndex.pop_back();
        }
    }

    int MjpegWriter::toJPGframe(const uchar * data, uint width, uint height, void *& pBuf)
    {
        const int req_comps = 3; // request BGR image, if (BGRA) req_comps = 4; 
        params param;
        param.m_quality = quality;
        param.m_subsampling = (subsampling_t)H2V2;
        param.m_color_space = (colorspace)InputColorSpace;
        int buf_size = width * height * 3; // allocate a buffer that's hopefully big enough (this is way overkill for jpeg)
        if (buf_size < 1024) buf_size = 1024;
        pBuf = malloc(buf_size);
        jpeg_encoder dst_image(param);
        if (!dst_image.compress_image_to_jpeg_file_in_memory(pBuf, buf_size, width, height, req_comps, data))
            return -1;

        return buf_size;
    }

    /////////////////////// jpeg_encoder ///////////////////

    jpeg_encoder::jpeg_encoder(params param) { m_params = param; }
        jpeg_encoder::~jpeg_encoder() { }

        class memory_stream : public output_stream
        {
            memory_stream(const memory_stream &);
            memory_stream &operator= (const memory_stream &);
            uchar *m_pBuf;
            uint m_buf_size, m_buf_ofs;

        public:
            memory_stream(void *pBuf, uint buf_size) : m_pBuf(static_cast<uchar*>(pBuf)), m_buf_size(buf_size), m_buf_ofs(0) { }
            virtual ~memory_stream() { }
            virtual bool put_buf(const void* pBuf, int len)
            {
                uint buf_remaining = m_buf_size - m_buf_ofs;
                if ((uint)len > buf_remaining)
                    return false;
                memcpy(m_pBuf + m_buf_ofs, pBuf, len);
                m_buf_ofs += len;
                return true;
            }
            uint get_size() const
            {
                return m_buf_ofs;
            }
        };

        bool jpeg_encoder::compress_image_to_jpeg_file_in_memory(void *&pDstBuf, int &buf_size, int width, int height, int num_channels, const uchar *pImage_data)
        {
            if (!init_clamp_table)
            {
                for (int i = -256; i < 512; i++)
                    clamp_table[i] = (uchar)(i < 0 ? 0 : i > 255 ? 255 : i);
            }

            if ((!pDstBuf) || (!buf_size))
                return false;

            memory_stream dst_stream(pDstBuf, buf_size);
            buf_size = 0;
            WriteImage(&dst_stream, pImage_data, width * num_channels, width, height, num_channels);
            buf_size = dst_stream.get_size();
            return true;
        }

    /////////////////////// GrFmtJpegWriter ///////////////////

    // FDCT with postscaling
    static void aan_fdct8x8(int *src, int *dst,
        int step, const int *postscale)
    {
        int  workspace[64], *work = workspace;
        int  i;

        // Pass 1: process rows
        for (i = 8; i > 0; i--, src += step, work += 8)
        {
            int x0 = src[0], x1 = src[7];
            int x2 = src[3], x3 = src[4];

            int x4 = x0 + x1; x0 -= x1;
            x1 = x2 + x3; x2 -= x3;

            work[7] = x0; work[1] = x2;
            x2 = x4 + x1; x4 -= x1;

            x0 = src[1]; x3 = src[6];
            x1 = x0 + x3; x0 -= x3;
            work[5] = x0;

            x0 = src[2]; x3 = src[5];
            work[3] = x0 - x3; x0 += x3;

            x3 = x0 + x1; x0 -= x1;
            x1 = x2 + x3; x2 -= x3;

            work[0] = x1; work[4] = x2;

            x0 = DCT_DESCALE((x0 - x4) * C0_707, fixb);
            x1 = x4 + x0; x4 -= x0;
            work[2] = x4; work[6] = x1;

            x0 = work[1]; x1 = work[3];
            x2 = work[5]; x3 = work[7];

            x0 += x1; x1 += x2; x2 += x3;
            x1 = DCT_DESCALE(x1*C0_707, fixb);

            x4 = x1 + x3; x3 -= x1;
            x1 = (x0 - x2)*C0_382;
            x0 = DCT_DESCALE(x0 * C0_541 + x1, fixb);
            x2 = DCT_DESCALE(x2 * C1_306 + x1, fixb);

            x1 = x0 + x3; x3 -= x0;
            x0 = x4 + x2; x4 -= x2;

            work[5] = x1; work[1] = x0;
            work[7] = x4; work[3] = x3;
        }

        work = workspace;
        // pass 2: process columns
        for (i = 8; i > 0; i--, work++, postscale += 8, dst += 8)
        {
            int  x0 = work[8 * 0], x1 = work[8 * 7];
            int  x2 = work[8 * 3], x3 = work[8 * 4];

            int  x4 = x0 + x1; x0 -= x1;
            x1 = x2 + x3; x2 -= x3;

            work[8 * 7] = x0; work[8 * 0] = x2;
            x2 = x4 + x1; x4 -= x1;

            x0 = work[8 * 1]; x3 = work[8 * 6];
            x1 = x0 + x3; x0 -= x3;
            work[8 * 4] = x0;

            x0 = work[8 * 2]; x3 = work[8 * 5];
            work[8 * 3] = x0 - x3; x0 += x3;

            x3 = x0 + x1; x0 -= x1;
            x1 = x2 + x3; x2 -= x3;

            dst[0] = DCT_DESCALE(x1 * postscale[0], postshift);
            dst[4] = DCT_DESCALE(x2 * postscale[4], postshift);

            x0 = DCT_DESCALE((x0 - x4)*C0_707, fixb);
            x1 = x4 + x0; x4 -= x0;

            dst[2] = DCT_DESCALE(x4 * postscale[2], postshift);
            dst[6] = DCT_DESCALE(x1 * postscale[6], postshift);

            x0 = work[8 * 0]; x1 = work[8 * 3];
            x2 = work[8 * 4]; x3 = work[8 * 7];

            x0 += x1; x1 += x2; x2 += x3;
            x1 = DCT_DESCALE(x1*C0_707, fixb);

            x4 = x1 + x3; x3 -= x1;
            x1 = (x0 - x2) * C0_382;
            x0 = DCT_DESCALE(x0 * C0_541 + x1, fixb);
            x2 = DCT_DESCALE(x2 * C1_306 + x1, fixb);

            x1 = x0 + x3; x3 -= x0;
            x0 = x4 + x2; x4 -= x2;

            dst[5] = DCT_DESCALE(x1 * postscale[5], postshift);
            dst[1] = DCT_DESCALE(x0 * postscale[1], postshift);
            dst[7] = DCT_DESCALE(x4 * postscale[7], postshift);
            dst[3] = DCT_DESCALE(x3 * postscale[3], postshift);
        }
    }

#if SSE
    static void SSE_aan_fdct8x8(short *src, int *dst, int step, const short *postscale)
    {
        const __m128i c0707 = _mm_set1_epi16(23167),
            c0541 = _mm_set1_epi16(17727),
            c1307 = _mm_set1_epi16(21414),
            c0383 = _mm_set1_epi16(12550),
            fix1 = _mm_set1_epi16(1),
            fix2 = _mm_set1_epi16(2);
        short dst_short[64];
        __m128i z = _mm_setzero_si128(), x0, x1, x2, x3, x4, tmp;
        __m128i work0, work1, work2, work3, work4, work5, work6, work7, // work
            tr_work0, tr_work1, tr_work2, tr_work3, tr_work4, tr_work5, tr_work6, tr_work7; // trans_work

        // Pass 1: process columns
        x0 = _mm_loadu_si128((const __m128i*)(src + 0 * step));
        x1 = _mm_loadu_si128((const __m128i*)(src + 7 * step));
        x2 = _mm_loadu_si128((const __m128i*)(src + 3 * step));
        x3 = _mm_loadu_si128((const __m128i*)(src + 4 * step));

        x4 = _mm_adds_epi16(x0, x1);   // x4 = x0 + x1
        x0 = _mm_subs_epi16(x0, x1);   // x0 -= x1

        x1 = _mm_adds_epi16(x2, x3);   // x1 = x2 + x3
        x2 = _mm_subs_epi16(x2, x3);   // x2 -= x3

        work7 = x0; work1 = x2;

        x2 = _mm_adds_epi16(x4, x1);   // x2 = x4 + x1
        x4 = _mm_subs_epi16(x4, x1);   // x4 -= x1

        x0 = _mm_loadu_si128((const __m128i*)(src + 1 * step));
        x3 = _mm_loadu_si128((const __m128i*)(src + 6 * step));

        x1 = _mm_adds_epi16(x0, x3);   // x1 = x0 + x3
        x0 = _mm_subs_epi16(x0, x3);   // x0 -= x3

        work5 = x0;

        x0 = _mm_loadu_si128((const __m128i*)(src + 2 * step));
        x3 = _mm_loadu_si128((const __m128i*)(src + 5 * step));

        work3 = _mm_sub_epi16(x0, x3);

        x0 = _mm_adds_epi16(x0, x3);   // x0 += x3
        x3 = _mm_adds_epi16(x0, x1);   // x3 = x0 + x1
        x0 = _mm_subs_epi16(x0, x1);   // x0 -= x1
        x1 = _mm_adds_epi16(x2, x3);   // x1 = x2 + x3
        x2 = _mm_subs_epi16(x2, x3);   // x2 -= x3

        work0 = x1; work4 = x2;

        x0 = _mm_mulhi_epi16(_mm_slli_epi16(_mm_adds_epi16(_mm_subs_epi16(x0, x4), fix1), 1), c0707); // DCT_DESCALE((x0 - x4) * C0_707, fixb);

        x1 = _mm_adds_epi16(x4, x0);   // x1 = x4 + x0
        x4 = _mm_subs_epi16(x4, x0);   // x4 -= x0

        work2 = x4; work6 = x1;

        x0 = work1; x1 = work3;
        x2 = work5; x3 = work7;

        x0 = _mm_adds_epi16(x0, x1);   // x0 += x1
        x1 = _mm_adds_epi16(x1, x2);   // x1 += x2
        x2 = _mm_adds_epi16(x2, x3);   // x2 += x3

        x1 = _mm_mulhi_epi16(_mm_adds_epi16(x1, x1), c0707); // DCT_DESCALE(x1 * C0_707, fixb);

        x4 = _mm_adds_epi16(x1, x3);   // x4 = x1 + x3
        x3 = _mm_subs_epi16(x3, x1);   // x3 -= x1

        x1 = _mm_mulhi_epi16(_mm_slli_epi16(_mm_adds_epi16(_mm_subs_epi16(x0, x2), fix2), 1), c0383);  // (x0 - x2) * C0_382;
        x0 = _mm_adds_epi16(_mm_mulhi_epi16(_mm_adds_epi16(_mm_adds_epi16(x0, x0), fix1), c0541), x1); // DCT_DESCALE(x0 * C0_541 + x1, fixb);
        x2 = _mm_adds_epi16(_mm_mulhi_epi16(_mm_adds_epi16(_mm_slli_epi16(x2, 2), fix1), c1307), x1);  // DCT_DESCALE(x2 * C1_306 + x1, fixb);

        x1 = _mm_adds_epi16(x0, x3);   // x1 = x0 + x3
        x3 = _mm_subs_epi16(x3, x0);   // x3 -= x0
        x0 = _mm_adds_epi16(x4, x2);   // x0 = x4 + x2
        x4 = _mm_subs_epi16(x4, x2);   // x4 -= x2

        work5 = x1; work1 = x0;
        work7 = x4; work3 = x3;

        // transform matrix

        tr_work0 = _mm_unpacklo_epi16(work0, work1);            // a0 b0 a1 b1 a2 b2 a3 b3
        tr_work1 = _mm_unpackhi_epi16(work0, work1);            // a4 b4 a5 b5 a6 b6 a7 b7
        tr_work2 = _mm_unpacklo_epi16(work2, work3);            // c0 d0 c1 d1 c2 d2 c3 d3
        tr_work3 = _mm_unpackhi_epi16(work2, work3);            // c4 d4 c5 d5 c6 d6 c7 d7
        tr_work4 = _mm_unpacklo_epi16(work4, work5);            // e0 f0 e1 f1 e2 f2 e3 f3
        tr_work5 = _mm_unpackhi_epi16(work4, work5);            // e4 f4 e5 f5 e6 f6 e7 f7
        tr_work6 = _mm_unpacklo_epi16(work6, work7);            // g0 h0 g1 h1 g2 h2 g3 h3
        tr_work7 = _mm_unpackhi_epi16(work6, work7);            // g4 h4 g5 h5 g6 h6 g7 h7

        work0    = _mm_unpacklo_epi32(tr_work0, tr_work2);      // a0 b0 c0 d0 a1 b1 c1 d1
        work1    = _mm_unpackhi_epi32(tr_work0, tr_work2);      // a2 b2 c2 d2 a3 b3 c3 d3
        work2    = _mm_unpacklo_epi32(tr_work1, tr_work3);      // a4 b4 c4 d4 a5 b5 c5 d5
        work3    = _mm_unpackhi_epi32(tr_work1, tr_work3);      // a6 b6 c6 d6 a7 b7 c7 d7
        work4    = _mm_unpacklo_epi32(tr_work4, tr_work6);      // e0 f0 g0 h0 e1 f1 g1 h1
        work5    = _mm_unpackhi_epi32(tr_work4, tr_work6);      // e2 f2 g2 h2 e3 f3 g3 h3
        work6    = _mm_unpacklo_epi32(tr_work5, tr_work7);      // e4 f4 g4 h4 e5 f5 g5 h5
        work7    = _mm_unpackhi_epi32(tr_work5, tr_work7);      // e6 f6 g6 h6 e7 f7 g7 h7

        tr_work0 = _mm_unpacklo_epi64(work0, work4);            // a0 b0 c0 d0 e0 f0 g0 h0
        tr_work1 = _mm_unpackhi_epi64(work0, work4);            // a1 b1 c1 d1 e1 f1 g1 h1
        tr_work2 = _mm_unpacklo_epi64(work1, work5);            // a0 b0 c0 d0 e0 f0 g0 h0
        tr_work3 = _mm_unpackhi_epi64(work1, work5);            // a1 b1 c1 d1 e1 f1 g1 h1
        tr_work4 = _mm_unpacklo_epi64(work2, work6);            // a0 b0 c0 d0 e0 f0 g0 h0
        tr_work5 = _mm_unpackhi_epi64(work2, work6);            // a1 b1 c1 d1 e1 f1 g1 h1
        tr_work6 = _mm_unpacklo_epi64(work3, work7);            // a0 b0 c0 d0 e0 f0 g0 h0
        tr_work7 = _mm_unpackhi_epi64(work3, work7);            // a1 b1 c1 d1 e1 f1 g1 h1

        // pass 2: process rows

        x0 = tr_work0, x1 = tr_work7;
        x2 = tr_work3, x3 = tr_work4;

        x4 = _mm_adds_epi16(x0, x1);   // x4 = x0 + x1
        x0 = _mm_subs_epi16(x0, x1);   // x0 -= x1

        x1 = _mm_adds_epi16(x2, x3);   // x1 = x2 + x3
        x2 = _mm_subs_epi16(x2, x3);   // x2 -= x3

        tr_work7 = x0; tr_work0 = x2;

        x2 = _mm_adds_epi16(x4, x1);   // x2 = x4 + x1
        x4 = _mm_subs_epi16(x4, x1);   // x4 -= x1

        x0 = tr_work1; x3 = tr_work6;

        x1 = _mm_adds_epi16(x0, x3);   // x1 = x0 + x3
        x0 = _mm_subs_epi16(x0, x3);   // x0 -= x3

        tr_work4 = x0;

        x0 = tr_work2; x3 = tr_work5;

        tr_work3 = _mm_sub_epi16(x0, x3);

        x0 = _mm_adds_epi16(x0, x3);   // x0 += x3
        x3 = _mm_adds_epi16(x0, x1);   // x3 = x0 + x1
        x0 = _mm_subs_epi16(x0, x1);   // x0 -= x1
        x1 = _mm_adds_epi16(x2, x3);   // x1 = x2 + x3
        x2 = _mm_subs_epi16(x2, x3);   // x2 -= x3

        tr_work1 = _mm_mulhi_epi16(x1, _mm_loadu_si128((const __m128i*)(postscale + 0 * 8))); // DCT_DESCALE(x1 * postscale[0], postshift);
        tr_work2 = _mm_mulhi_epi16(x2, _mm_loadu_si128((const __m128i*)(postscale + 4 * 8))); // DCT_DESCALE(x2 * postscale[4], postshift);
        _mm_storeu_si128((__m128i*)(dst_short + 0 * 8), tr_work1);
        _mm_storeu_si128((__m128i*)(dst_short + 4 * 8), tr_work2);

        x0 = _mm_mulhi_epi16(_mm_slli_epi16(_mm_subs_epi16(x0, x4), 1), c0707); // DCT_DESCALE((x0 - x4)*C0_707, fixb);

        x1 = _mm_adds_epi16(x4, x0);   // x1 = x4 + x0
        x4 = _mm_subs_epi16(x4, x0);   // x4 -= x0

        tr_work1 = _mm_mulhi_epi16(x4, _mm_loadu_si128((const __m128i*)(postscale + 2 * 8))); // DCT_DESCALE(x4 * postscale[2], postshift);
        tr_work2 = _mm_mulhi_epi16(x1, _mm_loadu_si128((const __m128i*)(postscale + 6 * 8))); // DCT_DESCALE(x1 * postscale[6], postshift);
        _mm_storeu_si128((__m128i*)(dst_short + 2 * 8), tr_work1);
        _mm_storeu_si128((__m128i*)(dst_short + 6 * 8), tr_work2);

        x0 = tr_work0; x1 = tr_work3;
        x2 = tr_work4; x3 = tr_work7;

        x0 = _mm_adds_epi16(x0, x1);   // x0 += x1
        x1 = _mm_adds_epi16(x1, x2);   // x1 += x2
        x2 = _mm_adds_epi16(x2, x3);   // x2 += x3

        x1 = _mm_mulhi_epi16(_mm_adds_epi16(x1, x1), c0707); // DCT_DESCALE(x1 * C0_707, fixb);

        x4 = _mm_adds_epi16(x1, x3);   // x4 = x1 + x3
        x3 = _mm_subs_epi16(x3, x1);   // x3 -= x1

        x1 = _mm_mulhi_epi16(_mm_slli_epi16(_mm_subs_epi16(x0, x2), 1), c0383); // (x0 - x2) * C0_382;

        x0 = _mm_adds_epi16(_mm_mulhi_epi16(_mm_adds_epi16(x0, x0), c0541), x1); // DCT_DESCALE(x0 * C0_541 + x1, fixb);
        x2 = _mm_adds_epi16(_mm_mulhi_epi16(_mm_slli_epi16(x2, 2), c1307), x1);  // DCT_DESCALE(x2 * C1_306 + x1, fixb);

        x1 = _mm_adds_epi16(x0, x3);   // x1 = x0 + x3
        x3 = _mm_subs_epi16(x3, x0);   // x3 -= x0
        x0 = _mm_adds_epi16(x4, x2);   // x0 = x4 + x2
        x4 = _mm_subs_epi16(x4, x2);   // x4 -= x2

        tr_work1 = _mm_mulhi_epi16(x1, _mm_loadu_si128((const __m128i*)(postscale + 5 * 8))); // DCT_DESCALE(x1 * postscale[5], postshift);
        tr_work2 = _mm_mulhi_epi16(x0, _mm_loadu_si128((const __m128i*)(postscale + 1 * 8))); // DCT_DESCALE(x0 * postscale[1], postshift);
        tr_work3 = _mm_mulhi_epi16(x4, _mm_loadu_si128((const __m128i*)(postscale + 7 * 8))); // DCT_DESCALE(x4 * postscale[7], postshift);
        tr_work4 = _mm_mulhi_epi16(x3, _mm_loadu_si128((const __m128i*)(postscale + 3 * 8))); // DCT_DESCALE(x3 * postscale[3], postshift);
        _mm_storeu_si128((__m128i*)(dst_short + 5 * 8), tr_work1);
        _mm_storeu_si128((__m128i*)(dst_short + 1 * 8), tr_work2);
        _mm_storeu_si128((__m128i*)(dst_short + 7 * 8), tr_work3);
        _mm_storeu_si128((__m128i*)(dst_short + 3 * 8), tr_work4);

        // for test only
        for (int k = 0; k < 64; k++)
        {
            *dst++ = (int)dst_short[k];
        }

    }

    #endif

    bool bsCreateEncodeHuffmanTable(const int* src, unsigned long* table, int max_size)
    {
        int  i, k;
        int  min_val = INT_MAX, max_val = INT_MIN;
        int  size;

        /* calc min and max values in the table */
        for (i = 1, k = 1; src[k] >= 0; i++)
        {
            int code_count = src[k++];

            for (code_count += k; k < code_count; k++)
            {
                int  val = src[k] > huff_val_shift;
                if (val < min_val)
                    min_val = val;
                if (val > max_val)
                    max_val = val;
            }
        }

        size = max_val - min_val + 3;

        if (size > max_size)
        {
            assert(0);
            return false;
        }

        memset(table, 0, size*sizeof(table[0]));

        table[0] = min_val;
        table[1] = size - 2;

        for (i = 1, k = 1; src[k] >= 0; i++)
        {
            int code_count = src[k++];

            for (code_count += k; k < code_count; k++)
            {
                int  val = src[k] >> huff_val_shift;
                int  code = src[k] & huff_code_mask;

                table[val - min_val + 2] = (code << 8) | i;
            }
        }
        return true;
    }

    int*  bsCreateSourceHuffmanTable(const uchar* src, int* dst,
        int max_bits, int first_bits)
    {
        int   i, val_idx, code = 0;
        int*  table = dst;
        *dst++ = first_bits;
        for (i = 1, val_idx = max_bits; i <= max_bits; i++)
        {
            int code_count = src[i - 1];
            dst[0] = code_count;
            code <<= 1;
            for (int k = 0; k < code_count; k++)
            {
                dst[k + 1] = (src[val_idx + k] << huff_val_shift) | (code + k);
            }
            code += code_count;
            dst += code_count + 1;
            val_idx += code_count;
        }
        dst[0] = -1;
        return  table;
    }

    bool jpeg_encoder::WriteImage(output_stream *pStream, const uchar* data, int step,
        int width, int height, int _channels)
    {
        assert(data && width > 0 && height > 0);
        WJpegBitStream  m_strm;
        if (!m_strm.Open(pStream)) return false;

        // encode the header and tables
        // for each mcu:
        //   convert rgb to yuv with downsampling (if color).
        //   for every block:
        //     calc dct and quantize
        //     encode block.
#if SSE
        short SSE_fdct_qtab[2][64];
#else
        int   fdct_qtab[2][64];
#endif
        int x, y;
        int i, j;
        const int max_quality = 3;
        int   quality = max_quality;
        WMByteStream& lowstrm = m_strm.m_low_strm;
        ulong huff_dc_tab[2][16];
        ulong huff_ac_tab[2][256];
        int  channels = _channels > 1 ? 3 : 1;
        int  x_scale = channels > 1 ? 2 : 1, y_scale = x_scale;
        int  dc_pred[] = { 0, 0, 0 };
        int  x_step = x_scale * 8;
        int  y_step = y_scale * 8;
        int  block[6][64];
        short  block_short[6][64];
        int  buffer[1024];
        int  luma_count = x_scale*y_scale;
        int  block_count = luma_count + channels - 1;
        int  Y_step = x_scale * 8;
        const int UV_step = 16;
        double inv_quality;

        if (quality < 3) quality = 3;
        if (quality > max_quality) quality = max_quality;

        inv_quality = 1. / quality;

        // Encode header
        lowstrm.PutBytes(jpegHeader, sizeof(jpegHeader)-1);

        // Encode quantization tables
        for (i = 0; i < (channels > 1 ? 2 : 1); i++)
        {
            const uchar* qtable = i == 0 ? jpegTableK1_T : jpegTableK2_T;
            int chroma_scale = i > 0 ? luma_count : 1;

            lowstrm.PutWord(0xffdb);   // DQT marker
            lowstrm.PutWord(2 + 65 * 1); // put single qtable
            lowstrm.PutByte(0 * 16 + i); // 8-bit table

            // put coefficients
            for (j = 0; j < 64; j++)
            {
                int idx = zigzag[j];
#if SSE
                int SSE_idx = SSE_zigzag[j],
                    SSE_qval = cvRound(qtable[SSE_idx] * inv_quality);
                SSE_qval = clamp_table[SSE_qval];
                SSE_fdct_qtab[i][SSE_idx] = (short)cvRound((1 << (postshift + 9)) /
                    (SSE_qval*chroma_scale*idct_prescale[idx]));
                lowstrm.PutByte(SSE_qval);
#else
                int qval = cvRound(qtable[idx] * inv_quality);
                qval = clamp_table[qval];
                fdct_qtab[i][idx] = cvRound((1 << (postshift + 9)) /
                    (qval*chroma_scale*idct_prescale[idx]));
                lowstrm.PutByte(qval);
#endif
            }
        }

        // Encode huffman tables
        for (i = 0; i < (channels > 1 ? 4 : 2); i++)
        {
            const uchar* htable = i == 0 ? jpegTableK3 : i == 1 ? jpegTableK5 :
                i == 2 ? jpegTableK4 : jpegTableK6;
            int is_ac_tab = i & 1;
            int idx = i >= 2;
            int tableSize = 16 + (is_ac_tab ? 162 : 12);

            lowstrm.PutWord(0xFFC4);      // DHT marker
            lowstrm.PutWord(3 + tableSize); // define one huffman table
            lowstrm.PutByte(is_ac_tab * 16 + idx); // put DC/AC flag and table index
            lowstrm.PutBytes(htable, tableSize); // put table

            bsCreateEncodeHuffmanTable(bsCreateSourceHuffmanTable(
                htable, buffer, 16, 9), is_ac_tab ? huff_ac_tab[idx] :
                huff_dc_tab[idx], is_ac_tab ? 256 : 16);
        }

        // put frame header
        lowstrm.PutWord(0xFFC0);          // SOF0 marker
        lowstrm.PutWord(8 + 3 * channels);  // length of frame header
        lowstrm.PutByte(8);               // sample precision
        lowstrm.PutWord(height);
        lowstrm.PutWord(width);
        lowstrm.PutByte(channels);        // number of components

        for (i = 0; i < channels; i++)
        {
            lowstrm.PutByte(i + 1);  // (i+1)-th component id (Y,U or V)
            if (i == 0)
                lowstrm.PutByte(x_scale * 16 + y_scale); // chroma scale factors
            else
                lowstrm.PutByte(1 * 16 + 1);
            lowstrm.PutByte(i > 0); // quantization table idx
        }

        // put scan header
        lowstrm.PutWord(0xFFDA);          // SOS marker
        lowstrm.PutWord(6 + 2 * channels);  // length of scan header
        lowstrm.PutByte(channels);        // number of components in the scan

        for (i = 0; i < channels; i++)
        {
            lowstrm.PutByte(i + 1);             // component id
            lowstrm.PutByte((i>0) * 16 + (i > 0));// selection of DC & AC tables
        }

        lowstrm.PutWord(0 * 256 + 63);// start and end of spectral selection - for
        // sequental DCT start is 0 and end is 63

        lowstrm.PutByte(0);  // successive approximation bit position 
        // high & low - (0,0) for sequental DCT  

        switch (m_params.m_color_space)
        {
        case COLORSPACE_BGR:
            break;
        case COLORSPACE_YUV444P:
            step /= _channels;
            break;
        }

        // encode data
        for (y = 0; y < height; y += y_step, data += y_step*step)
        {
            for (x = 0; x < width; x += x_step)
            {
                int x_limit = x_step;
                int y_limit = y_step;
                int* Y_data = block[0];
                const uchar* rgb_data = data + x*_channels;

                if (x + x_limit > width) x_limit = width - x;
                if (y + y_limit > height) y_limit = height - y;

                memset(block, 0, block_count * 64 * sizeof(block[0][0]));
                memset(block_short, 0, block_count * 64 * sizeof(block_short[0][0]));

                if (channels > 1)
                {
                    switch (m_params.m_color_space)
                    {
                    case COLORSPACE_BGR:
                        {
#if SSE
                            short* Y_data = block_short[0];
                            short* UV_data = block_short[luma_count];
                            __m128i m0 = _mm_setr_epi16(0, m00, m01, m02, m00, m01, m02, 0);
                            __m128i m1 = _mm_setr_epi16(0, m10, m11, m12, m10, m11, m12, 0);
                            __m128i m2 = _mm_setr_epi16(0, m20, m21, m22, m20, m21, m22, 0);
                            __m128i m3_all = _mm_setr_epi32(Hm, Hm, Hm, Hm), m3_all_1 = _mm_setr_epi32(Hm, 0, Hm, 0),
                                m3_all_2 = _mm_setr_epi32(0, Hm, 0, Hm);
                            __m128i z = _mm_setzero_si128(), t0, t1, t2, t0_2, t_mid, r0, r1, v0, v1, v2, v3, v0_2, v1_2, v2_2, v3_2,
                                shiftY = _mm_set1_epi16(512);
                            const int SSE_step = 8;
                            if ((y_limit == 16) && (x_limit == 16)) 
                            for (i = 0; i < y_limit; i += 2, rgb_data += step * 2, Y_data += Y_step, UV_data += SSE_step)
                            {
                                for (j = 0; j < x_limit; j += SSE_step, rgb_data += _channels * SSE_step, Y_data += SSE_step, UV_data += 4)
                                {
                                    // 1st row
                                    v0 = _mm_loadl_epi64((const __m128i*)(rgb_data));
                                    v1 = _mm_loadl_epi64((const __m128i*)(rgb_data + 8));
                                    v2 = _mm_loadl_epi64((const __m128i*)(rgb_data + 16));
                                    v0 = _mm_unpacklo_epi8(v0, z); // b0 g0 r0 b1 g1 r1 b2 g2
                                    v1 = _mm_unpacklo_epi8(v1, z); // r2 b3 g3 r3 b4 g4 r4 b5
                                    v2 = _mm_unpacklo_epi8(v2, z); // g5 r5 b6 g6 r6 b7 g7 r7

                                    v3 = _mm_srli_si128(v2, 2); // ? b6 g6 r6 b7 g7 r7 0
                                    v2 = _mm_or_si128(_mm_slli_si128(v2, 10), _mm_srli_si128(v1, 6)); // ? b4 g4 r4 b5 g5 r5 ?
                                    v1 = _mm_or_si128(_mm_slli_si128(v1, 6), _mm_srli_si128(v0, 10)); // ? b2 g2 r2 b3 g3 r3 ?
                                    v0 = _mm_slli_si128(v0, 2); // 0 b0 g0 r0 b1 g1 r1 ?

                                    // 2nd row
                                    v0_2 = _mm_loadl_epi64((const __m128i*)(rgb_data + step));
                                    v1_2 = _mm_loadl_epi64((const __m128i*)(rgb_data + step + 8));
                                    v2_2 = _mm_loadl_epi64((const __m128i*)(rgb_data + step + 16));
                                    v0_2 = _mm_unpacklo_epi8(v0_2, z); // b0 g0 r0 b1 g1 r1 b2 g2
                                    v1_2 = _mm_unpacklo_epi8(v1_2, z); // r2 b3 g3 r3 b4 g4 r4 b5
                                    v2_2 = _mm_unpacklo_epi8(v2_2, z); // g5 r5 b6 g6 r6 b7 g7 r7

                                    v3_2 = _mm_srli_si128(v2_2, 2); // ? b6 g6 r6 b7 g7 r7 0
                                    v2_2 = _mm_or_si128(_mm_slli_si128(v2_2, 10), _mm_srli_si128(v1_2, 6)); // ? b4 g4 r4 b5 g5 r5 ?
                                    v1_2 = _mm_or_si128(_mm_slli_si128(v1_2, 6), _mm_srli_si128(v0_2, 10)); // ? b2 g2 r2 b3 g3 r3 ?
                                    v0_2 = _mm_slli_si128(v0_2, 2); // 0 b0 g0 r0 b1 g1 r1 ?

                                    // process pixels 0 & 1 (1st and 2nd rows)

                                    t0 = _mm_madd_epi16(v0, m2);      // a0 b0 a1 b1         : Y0_1 Y1_1
                                    t0_2 = _mm_madd_epi16(v0_2, m2);  // a0_2 b0_2 a1_2 b1_2 : Y0_2 Y1_2
                                    t_mid = _mm_adds_epi16(v0, v0_2);
                                    t1 = _mm_madd_epi16(t_mid, m1);      // c0 d0 c1 d1 : U0_1 + U0_2  U1_1 + U1_2
                                    t2 = _mm_madd_epi16(t_mid, m0);      // e0 f0 e1 f1 : V0_1 + V0_2  V1_1 + V1_2

                                    v0 = _mm_unpacklo_epi32(t0, t0_2); // a0 a0_2 b0 b0_2
                                    t0 = _mm_unpackhi_epi32(t0, t0_2); // a1 a1_2 b1 b1_2

                                    r0 = _mm_unpacklo_epi64(t1, t2);  // c0 d0 e0 f0
                                    t2 = _mm_unpackhi_epi64(t1, t2);  // c1 d1 e1 f1
                                    t1 = _mm_add_epi32(r0, t2);       // c  d  e  f : U0_1 + U0_2 + U1_1 + U1_2  V0_1 + V0_2 + V1_1 + V1_2
                                    t2 = _mm_unpacklo_epi32(t1, z);   // c 0 d 0
                                    t1 = _mm_unpackhi_epi32(t1, z);   // e 0 f 0

                                    r0 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi32(v0, t0), _mm_unpackhi_epi32(v0, t0)), m3_all); // Y0_1 Y1_1 Y0_2 Y1_2
                                    r1 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(t2, t1), _mm_unpackhi_epi64(t2, t1)), m3_all_1); // U0 0 V0 0
                                    r0 = _mm_srai_epi32(r0, BITS - 2);
                                    r1 = _mm_srai_epi32(r1, BITS - 2);
                                    v0 = _mm_packs_epi32(r0, r1); // Y0_1 Y1_1 Y0_2 Y1_2 U0 0 V0 0

                                    // process pixels 2 & 3 (1st and 2nd rows)
                                    t0 = _mm_madd_epi16(v1, m2);      // a0 b0 a1 b1         : Y2_1 Y3_1
                                    t0_2 = _mm_madd_epi16(v1_2, m2);  // a0_2 b0_2 a1_2 b1_2 : Y2_2 Y3_2
                                    t_mid = _mm_adds_epi16(v1, v1_2);
                                    t1 = _mm_madd_epi16(t_mid, m1);      // c0 d0 c1 d1 : U2_1 + U2_2  U3_1 + U3_2
                                    t2 = _mm_madd_epi16(t_mid, m0);      // e0 f0 e1 f1 : V2_1 + V2_2  V3_1 + V3_2

                                    v1 = _mm_unpacklo_epi32(t0, t0_2); // a0 a0_2 b0 b0_2
                                    t0 = _mm_unpackhi_epi32(t0, t0_2); // a1 a1_2 b1 b1_2

                                    r0 = _mm_unpacklo_epi64(t1, t2);  // c0 d0 e0 f0
                                    t2 = _mm_unpackhi_epi64(t1, t2);  // c1 d1 e1 f1
                                    t1 = _mm_add_epi32(r0, t2);       // c  d  e  f : U2_1 + U2_2 + U3_1 + U3_2  V2_1 + V2_2 + V3_1 + V3_2
                                    t2 = _mm_unpacklo_epi32(t1, z);   // c 0 d 0
                                    t1 = _mm_unpackhi_epi32(t1, z);   // e 0 f 0

                                    r0 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi32(v1, t0), _mm_unpackhi_epi32(v1, t0)), m3_all); // Y2_1 Y3_1 Y2_2 Y3_2
                                    r1 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(t2, t1), _mm_unpackhi_epi64(t2, t1)), m3_all_1); // U1 0 V1 0
                                    r0 = _mm_srai_epi32(r0, BITS - 2);
                                    r1 = _mm_srai_epi32(r1, BITS - 2);
                                    v1 = _mm_packs_epi32(r0, r1); // Y2_1 Y3_1 Y2_2 Y3_2 U1 0 V1 0

                                    r0 = _mm_unpacklo_epi32(v0, v1); // Y0_1 Y1_1 Y2_1 Y3_1 Y0_2 Y1_2 Y2_2 Y3_2
                                    v1 = _mm_unpackhi_epi16(v0, v1); // U0 U1 0 0 V0 V1 0 0
                                    v0 = r0;

                                    //  process pixels 4 & 5 (1st and 2nd rows)
                                    t0 = _mm_madd_epi16(v2, m2);      // a0 b0 a1 b1         : Y4_1 Y5_1
                                    t0_2 = _mm_madd_epi16(v2_2, m2);  // a0_2 b0_2 a1_2 b1_2 : Y4_2 Y5_2
                                    t_mid = _mm_adds_epi16(v2, v2_2);
                                    t1 = _mm_madd_epi16(t_mid, m1);      // c0 d0 c1 d1 : U4_1 + U4_2  U5_1 + U5_2
                                    t2 = _mm_madd_epi16(t_mid, m0);      // e0 f0 e1 f1 : V4_1 + V4_2  V5_1 + V5_2

                                    v2 = _mm_unpacklo_epi32(t0, t0_2); // a0 a0_2 b0 b0_2
                                    t0 = _mm_unpackhi_epi32(t0, t0_2); // a1 a1_2 b1 b1_2

                                    r0 = _mm_unpacklo_epi64(t1, t2);  // c0 d0 e0 f0
                                    t2 = _mm_unpackhi_epi64(t1, t2);  // c1 d1 e1 f1
                                    t1 = _mm_add_epi32(r0, t2);       // c  d  e  f : U4_1 + U4_2 + U5_1 + U5_2  V4_1 + V4_2 + V5_1 + V5_2
                                    t2 = _mm_unpacklo_epi32(z, t1);   // 0 c 0 d
                                    t1 = _mm_unpackhi_epi32(z, t1);   // 0 e 0 f

                                    r0 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi32(v2, t0), _mm_unpackhi_epi32(v2, t0)), m3_all); // Y4_1 Y5_1 Y4_2 Y5_2
                                    r1 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(t2, t1), _mm_unpackhi_epi64(t2, t1)), m3_all_2); // 0 U2 0 V2
                                    r0 = _mm_srai_epi32(r0, BITS - 2);
                                    r1 = _mm_srai_epi32(r1, BITS - 2);
                                    v2 = _mm_packs_epi32(r0, r1); // Y4_1 Y5_1 Y4_2 Y5_2 0 U2 0 V2

                                    // proce ss pixels 6 & 7 (1st and 2nd rows)
                                    t0 = _mm_madd_epi16(v3, m2);      // a0 b0 a1 b1         : Y6_1 Y7_1
                                    t0_2 = _mm_madd_epi16(v3_2, m2);  // a0_2 b0_2 a1_2 b1_2 : Y6_2 Y7_2
                                    t_mid = _mm_adds_epi16(v3, v3_2);
                                    t1 = _mm_madd_epi16(t_mid, m1);      // c0 d0 c1 d1 : U6_1 + U6_2  U7_1 + U7_2
                                    t2 = _mm_madd_epi16(t_mid, m0);      // e0 f0 e1 f1 : V6_1 + V6_2  V7_1 + V7_2

                                    v3 = _mm_unpacklo_epi32(t0, t0_2); // a0 a0_2 b0 b0_2
                                    t0 = _mm_unpackhi_epi32(t0, t0_2); // a1 a1_2 b1 b1_2

                                    r0 = _mm_unpacklo_epi64(t1, t2);  // c0 d0 e0 f0
                                    t2 = _mm_unpackhi_epi64(t1, t2);  // c1 d1 e1 f1
                                    t1 = _mm_add_epi32(r0, t2);       // c  d  e  f : U6_1 + U7_2 + U7_1 + U7_2  V6_1 + V6_2 + V7_1 + V7_2
                                    t2 = _mm_unpacklo_epi32(z, t1);   // 0 c 0 d
                                    t1 = _mm_unpackhi_epi32(z, t1);   // 0 e 0 f

                                    r0 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi32(v3, t0), _mm_unpackhi_epi32(v3, t0)), m3_all); // Y6_1 Y7_1 Y6_2 Y7_2
                                    r1 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(t2, t1), _mm_unpackhi_epi64(t2, t1)), m3_all_2); // 0 U3 0 V3
                                    r0 = _mm_srai_epi32(r0, BITS - 2);
                                    r1 = _mm_srai_epi32(r1, BITS - 2);
                                    v3 = _mm_packs_epi32(r0, r1); // Y6_1 Y7_1 Y6_2 Y7_2 0 U3 0 V3

                                    r0 = _mm_unpacklo_epi32(v2, v3); // Y4_1 Y5_1 Y6_1 Y7_1 Y4_2 Y5_2 Y6_2 Y7_2
                                    v3 = _mm_unpackhi_epi16(v2, v3); // 0 0 U2 U3 0 0 V2 V3
                                    v2 = r0;

                                    // Y record
                                    t2 = _mm_unpacklo_epi64(v0, v2); // Y0_1 ... Y7_1
                                    t2 = _mm_subs_epi16(t2, shiftY);
                                    _mm_storeu_si128((__m128i*)(Y_data), t2);
                                    t2 = _mm_unpackhi_epi64(v0, v2); // Y0_2 ... Y7_2
                                    t2 = _mm_subs_epi16(t2, shiftY);
                                    _mm_storeu_si128((__m128i*)(Y_data + 16), t2);

                                    // U, V record
                                    t1 = _mm_adds_epi16(v1, v3);     // U0 U1 U2 U3 V0 V1 V2 V3
                                    _mm_storel_epi64((__m128i*)UV_data, t1); // store U0 U1 U2 U3
                                    t1 = _mm_srli_si128(t1, 8);      // V0 V1 V2 V3 0 0 0 0
                                    _mm_storel_epi64((__m128i*)(UV_data + 8), t1); // store V0 V1 V2 V3
                                }

                                rgb_data -= x_limit*_channels;

                            }
                            else
                            {
                                for (i = 0; i < y_limit; i++, rgb_data += step, Y_data += Y_step)
                                {
                                    for (j = 0; j < x_limit; j++, rgb_data += _channels)
                                    {
                                        int r = rgb_data[2];
                                        int g = rgb_data[1];
                                        int b = rgb_data[0];

                                        int Y = DCT_DESCALE(r*y_r + g*y_g + b*y_b, fixc - 2) - 128 * 4;
                                        int U = DCT_DESCALE(r*cb_r + g*cb_g + b*cb_b, fixc - 2);
                                        int V = DCT_DESCALE(r*cr_r + g*cr_g + b*cr_b, fixc - 2);

                                        int j2 = j >> (x_scale - 1);

                                        Y_data[j] = (short)Y;
                                        UV_data[j2] += (short)U;
                                        UV_data[j2 + 8] += (short)V;
                                    }
                                    rgb_data -= x_limit*_channels;
                                    if (((i + 1) & (y_scale - 1)) == 0)
                                    {
                                        UV_data += UV_step;
                                    }
                                }
                            }
                            // for test only
                            for (int k = 0; k < 64; k++)
                            {
                                block[0][k] = (int)block_short[0][k];
                                block[1][k] = (int)block_short[1][k];
                                block[2][k] = (int)block_short[2][k];
                                block[3][k] = (int)block_short[3][k];
                                block[4][k] = (int)block_short[4][k];
                                block[5][k] = (int)block_short[5][k];
                            }
#else
                            int* UV_data = block[luma_count];
                            for (i = 0; i < y_limit; i++, rgb_data += step, Y_data += Y_step)
                            {
                                for (j = 0; j < x_limit; j++, rgb_data += _channels)
                                {
                                    int r = rgb_data[2];
                                    int g = rgb_data[1];
                                    int b = rgb_data[0];

                                    int Y = DCT_DESCALE(r*y_r + g*y_g + b*y_b, fixc - 2) - 128 * 4;
                                    int U = DCT_DESCALE(r*cb_r + g*cb_g + b*cb_b, fixc - 2);
                                    int V = DCT_DESCALE(r*cr_r + g*cr_g + b*cr_b, fixc - 2);

                                    int j2 = j >> (x_scale - 1);

                                    Y_data[j] = Y;
                                    UV_data[j2] += U;
                                    UV_data[j2 + 8] += V;
                                }
                                rgb_data -= x_limit * _channels;
                                if (((i + 1) & (y_scale - 1)) == 0)
                                {
                                    UV_data += UV_step;
                                }
                            }
#endif
                        }
                        break;
                        case COLORSPACE_YUV444P:
                        {
                            const uchar* rgb_data = data + x;
                            const int row_step = 2, channel_step = width * height;
                            int* UV_data = block[luma_count];
                            for (i = 0; i < y_limit; i++, rgb_data += step, Y_data += Y_step)
                            {
                                for (j = 0; j < x_limit; j++)
                                {
                                    Y_data[j] = (*(rgb_data + j) - 128) << 2;
                                }
                                if ((i & 1) == 0)
                                {
                                    for (j = 0; j < x_limit; j += 2)
                                    {
                                        int j2 = j >> (x_scale - 1);

                                        UV_data[j2 + 8] = (*(rgb_data + channel_step + j) - 128) << 4;
                                        UV_data[j2]     = (*(rgb_data + channel_step * 2 + j) - 128) << 4;
                                    }
                                    UV_data += UV_step;
                                }
                            }
                        }
                        break;
                    }
                }
                else
                {
                    for (i = 0; i < y_limit; i++, rgb_data += step, Y_data += Y_step)
                    {
                        for (j = 0; j < x_limit; j++)
                            Y_data[j] = rgb_data[j] * 4 - 128 * 4;
                    }
                }

                for (i = 0; i < block_count; i++)
                {
                    int is_chroma = i >= luma_count;
                    int src_step = x_scale * 8;
                    int run = 0, val;
#if SSE
                    short* src_ptr = block_short[i & -2] + (i & 1) * 8;
                    const ulong* htable = huff_ac_tab[is_chroma];
                    SSE_aan_fdct8x8(src_ptr, buffer, src_step, SSE_fdct_qtab[is_chroma]);
#else
                    int* src_ptr = block[i & -2] + (i & 1) * 8;
                    const ulong* htable = huff_ac_tab[is_chroma];
                    aan_fdct8x8(src_ptr, buffer, src_step, fdct_qtab[is_chroma]);
                    
#endif
                    {
                        j = is_chroma + (i > luma_count);
                        val = buffer[0] - dc_pred[j];
                        dc_pred[j] = buffer[0];
                        float a = (float)val;
                        int cat = (((int&)a >> 23) & 255) - (126 & (val ? -1 : 0));

                        assert(cat <= 11);
                        m_strm.PutHuff(cat, huff_dc_tab[is_chroma]);
                        m_strm.Put(val - (val < 0 ? 1 : 0), cat);
                    }

                    for (j = 1; j < 64; j++)
                    {
#if SSE
                        val = buffer[SSE_zigzag[j]];
#else
                        val = buffer[zigzag[j]];
#endif
                        if (val == 0)
                        {
                            run++;
                        }
                        else
                        {
                            while (run >= 16)
                            {
                                m_strm.PutHuff(0xF0, htable); // encode 16 zeros
                                run -= 16;
                            }

                            {
                                float a = (float)val;
                                int cat = (((int&)a >> 23) & 255) - (126 & (val ? -1 : 0));

                                assert(cat <= 10);
                                m_strm.PutHuff(cat + run * 16, htable);
                                m_strm.Put(val - (val < 0 ? 1 : 0), cat);
                            }

                            run = 0;
                        }
                    }

                    if (run)
                    {
                        m_strm.PutHuff(0x00, htable); // encode EOB
                    }
                }
            }
        }

        // Flush 
        m_strm.Flush();

        lowstrm.PutWord(0xFFD9); // EOI marker
        m_strm.Close();

        return true;
    }

#define  BS_DEF_BLOCK_SIZE   (1<<15)

    const ulong bs_bit_mask[] = {
        0,
        0x00000001, 0x00000003, 0x00000007, 0x0000000F,
        0x0000001F, 0x0000003F, 0x0000007F, 0x000000FF,
        0x000001FF, 0x000003FF, 0x000007FF, 0x00000FFF,
        0x00001FFF, 0x00003FFF, 0x00007FFF, 0x0000FFFF,
        0x0001FFFF, 0x0003FFFF, 0x0007FFFF, 0x000FFFFF,
        0x001FFFFF, 0x003FFFFF, 0x007FFFFF, 0x00FFFFFF,
        0x01FFFFFF, 0x03FFFFFF, 0x07FFFFFF, 0x0FFFFFFF,
        0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF
    };

    void bsBSwapBlock(uchar *start, uchar *end)
    {
        ulong* data = (ulong*)start;
        int i, size = (int)(end - start + 3) / 4;

        for (i = 0; i < size; i++)
        {
            ulong temp = data[i];
            temp = BSWAP(temp);
            data[i] = temp;
        }
    }

    bool  bsIsBigEndian(void)
    {
        return (((const int*)"\0\x1\x2\x3\x4\x5\x6\x7")[0] & 255) != 0;
    }

    bool bsCreateDecodeHuffmanTable(const int* src, short* table, int max_size)
    {
        const int forbidden_entry = (RBS_HUFF_FORB << 4) | 1;
        int       first_bits = src[0];
        struct
        {
            int bits;
            int offset;
        }
        sub_tables[1 << 11];
        int  size = (1 << first_bits) + 1;
        int  i, k;

        /* calc bit depths of sub tables */
        memset(sub_tables, 0, ((size_t)1 << first_bits)*sizeof(sub_tables[0]));
        for (i = 1, k = 1; src[k] >= 0; i++)
        {
            int code_count = src[k++];
            int sb = i - first_bits;

            if (sb <= 0)
                k += code_count;
            else
            for (code_count += k; k < code_count; k++)
            {
                int  code = src[k] & huff_code_mask;
                sub_tables[code >> sb].bits = sb;
            }
        }

        /* calc offsets of sub tables and whole size of table */
        for (i = 0; i < (1 << first_bits); i++)
        {
            int b = sub_tables[i].bits;
            if (b > 0)
            {
                b = 1 << b;
                sub_tables[i].offset = size;
                size += b + 1;
            }
        }

        if (size > max_size)
        {
            assert(0);
            return false;
        }

        /* fill first table and subtables with forbidden values */
        for (i = 0; i < size; i++)
        {
            table[i] = (short)forbidden_entry;
        }

        /* write header of first table */
        table[0] = (short)first_bits;

        /* fill first table and sub tables */
        for (i = 1, k = 1; src[k] >= 0; i++)
        {
            int code_count = src[k++];
            for (code_count += k; k < code_count; k++)
            {
                int  table_bits = first_bits;
                int  code_bits = i;
                int  code = src[k] & huff_code_mask;
                int  val = src[k] >> huff_val_shift;
                int  j, offset = 0;

                if (code_bits > table_bits)
                {
                    int idx = code >> (code_bits -= table_bits);
                    code &= (1 << code_bits) - 1;
                    offset = sub_tables[idx].offset;
                    table_bits = sub_tables[idx].bits;
                    /* write header of subtable */
                    table[offset] = (short)table_bits;
                    /* write jump to subtable */
                    table[idx + 1] = (short)(offset << 4);
                }

                table_bits -= code_bits;
                assert(table_bits >= 0);
                val = (val << 4) | code_bits;
                offset += (code << table_bits) + 1;

                for (j = 0; j < (1 << table_bits); j++)
                {
                    assert(table[offset + j] == forbidden_entry);
                    table[offset + j] = (short)val;
                }
            }
        }
        return true;
    }


    /////////////////////////// WBaseStream /////////////////////////////////

    // WBaseStream - base class for output streams
    WBaseStream::WBaseStream()
    {
        m_start = m_end = m_current = 0;
        m_stream = 0;
        m_block_size = BS_DEF_BLOCK_SIZE;
        m_is_opened = false;
    }


    WBaseStream::~WBaseStream()
    {
        Close();    // Close files
        Release();  // free  buffers
    }


    bool  WBaseStream::IsOpened()
    {
        return m_is_opened;
    }


    void  WBaseStream::Allocate()
    {
        if (!m_start)
            m_start = new uchar[m_block_size];

        m_end = m_start + m_block_size;
        m_current = m_start;
    }


    void  WBaseStream::WriteBlock()
    {
        int size = (int)(m_current - m_start);
        m_stream->put_buf(m_start, size);
        m_current = m_start;
        m_block_pos += size;
    }


    bool  WBaseStream::Open(output_stream *stream)
    {
        Close();
        Allocate();

        if (stream)
        {
            m_stream = stream;
            m_is_opened = true;
            m_block_pos = 0;
            m_current = m_start;
        }
        return m_stream != 0;
    }


    void  WBaseStream::Close()
    {
        if (m_stream)
        {
            WriteBlock();
            m_stream = 0;
        }
        m_is_opened = false;
    }


    void  WBaseStream::Release()
    {
        if (m_start)
        {
            delete[] m_start;
        }
        m_start = m_end = m_current = 0;
    }


    void  WBaseStream::SetBlockSize(int block_size)
    {
        assert(block_size > 0 && (block_size & (block_size - 1)) == 0);

        if (m_start && block_size == m_block_size) return;
        Release();
        m_block_size = block_size;
        Allocate();
    }


    int  WBaseStream::GetPos()
    {
        assert(IsOpened());
        return m_block_pos + (int)(m_current - m_start);
    }


    ///////////////////////////// WLByteStream /////////////////////////////////// 

    WLByteStream::~WLByteStream()
    {
    }

    void WLByteStream::PutByte(int val)
    {
        *m_current++ = (uchar)val;
        if (m_current >= m_end)
            WriteBlock();
    }


    void WLByteStream::PutBytes(const void* buffer, int count)
    {
        uchar* data = (uchar*)buffer;

        assert(data && m_current && count >= 0);

        while (count)
        {
            int l = (int)(m_end - m_current);

            if (l > count)
                l = count;

            if (l > 0)
            {
                memcpy(m_current, data, l);
                m_current += l;
                data += l;
                count -= l;
            }
            if (m_current == m_end)
                WriteBlock();
        }
    }


    void WLByteStream::PutWord(int val)
    {
        uchar *current = m_current;

        if (current + 1 < m_end)
        {
            current[0] = (uchar)val;
            current[1] = (uchar)(val >> 8);
            m_current = current + 2;
            if (m_current == m_end)
                WriteBlock();
        }
        else
        {
            PutByte(val);
            PutByte(val >> 8);
        }
    }


    void WLByteStream::PutDWord(int val)
    {
        uchar *current = m_current;

        if (current + 3 < m_end)
        {
            current[0] = (uchar)val;
            current[1] = (uchar)(val >> 8);
            current[2] = (uchar)(val >> 16);
            current[3] = (uchar)(val >> 24);
            m_current = current + 4;
            if (m_current == m_end)
                WriteBlock();
        }
        else
        {
            PutByte(val);
            PutByte(val >> 8);
            PutByte(val >> 16);
            PutByte(val >> 24);
        }
    }


    ///////////////////////////// WMByteStream /////////////////////////////////// 

    WMByteStream::~WMByteStream()
    {
    }


    void WMByteStream::PutWord(int val)
    {
        uchar *current = m_current;

        if (current + 1 < m_end)
        {
            current[0] = (uchar)(val >> 8);
            current[1] = (uchar)val;
            m_current = current + 2;
            if (m_current == m_end)
                WriteBlock();
        }
        else
        {
            PutByte(val >> 8);
            PutByte(val);
        }
    }


    void WMByteStream::PutDWord(int val)
    {
        uchar *current = m_current;

        if (current + 3 < m_end)
        {
            current[0] = (uchar)(val >> 24);
            current[1] = (uchar)(val >> 16);
            current[2] = (uchar)(val >> 8);
            current[3] = (uchar)val;
            m_current = current + 4;
            if (m_current == m_end)
                WriteBlock();
        }
        else
        {
            PutByte(val >> 24);
            PutByte(val >> 16);
            PutByte(val >> 8);
            PutByte(val);
        }
    }


    ///////////////////////////// WMBitStream /////////////////////////////////// 

    WMBitStream::WMBitStream()
    {
        m_pad_val = 0;
        ResetBuffer();
    }


    WMBitStream::~WMBitStream()
    {
    }


    bool  WMBitStream::Open(output_stream *stream)
    {
        ResetBuffer();
        return WBaseStream::Open(stream);
    }


    void  WMBitStream::ResetBuffer()
    {
        m_val = 0;
        m_bit_idx = 32;
        m_current = m_start;
    }

    void  WMBitStream::Flush()
    {
        if (m_bit_idx < 32)
        {
            Put(m_pad_val, m_bit_idx & 7);
            *((ulong*&)m_current)++ = m_val;
        }
    }


    void  WMBitStream::Close()
    {
        if (m_is_opened)
        {
            Flush();
            WBaseStream::Close();
        }
    }


    void  WMBitStream::WriteBlock()
    {
        if (!bsIsBigEndian())
            bsBSwapBlock(m_start, m_current);
        WBaseStream::WriteBlock();
    }


    int  WMBitStream::GetPos()
    {
        return WBaseStream::GetPos() + ((32 - m_bit_idx) >> 3);
    }


    void  WMBitStream::Put(int val, int bits)
    {
        int  bit_idx = m_bit_idx - bits;
        ulong  curval = m_val;

        assert(0 <= bits && bits < 32);

        val &= bs_bit_mask[bits];

        if (bit_idx >= 0)
        {
            curval |= val << bit_idx;
        }
        else
        {
            *((ulong*&)m_current)++ = curval | ((unsigned)val >> -bit_idx);
            if (m_current >= m_end)
            {
                WriteBlock();
            }
            bit_idx += 32;
            curval = val << bit_idx;
        }

        m_val = curval;
        m_bit_idx = bit_idx;
    }


    void  WMBitStream::PutHuff(int val, const ulong* table)
    {
        int min_val = (int)table[0];
        val -= min_val;

        //assert((unsigned)val < table[1]);

        ulong code = table[val + 2];
        assert(code != 0);

        Put(code >> 8, code & 255);
    }

    ////////////////////// WJpegStream ///////////////////////

    WJpegBitStream::WJpegBitStream()
    {
    }


    WJpegBitStream::~WJpegBitStream()
    {
        Close();
        m_is_opened = false;
    }



    bool  WJpegBitStream::Open(output_stream *stream)
    {
        Close();
        Allocate();

        m_is_opened = m_low_strm.Open(stream);
        if (m_is_opened)
        {
            m_block_pos = 0;
            ResetBuffer();
        }
        return m_is_opened;
    }


    void  WJpegBitStream::Close()
    {
        if (m_is_opened)
        {
            Flush();
            m_low_strm.Close();
            m_is_opened = false;
        }
    }


    void  WJpegBitStream::Flush()
    {
        Put(-1, m_bit_idx & 31);
        *((ulong*&)m_current)++ = m_val;
        WriteBlock();
        ResetBuffer();
    }


    void  WJpegBitStream::WriteBlock()
    {
        uchar* ptr = m_start;
        if (!bsIsBigEndian())
            bsBSwapBlock(m_start, m_current);

        while (ptr < m_current)
        {
            int val = *ptr++;
            m_low_strm.PutByte(val);
            if (val == 0xff)
            {
                m_low_strm.PutByte(0);
            }
        }

        m_current = m_start;
    }
}
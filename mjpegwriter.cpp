
#include "mjpegwriter.hpp"
#include "opencv2/core/utility.hpp"
#include <smmintrin.h>

namespace jcodec{

#define SSE 1

#define fourCC(a,b,c,d) ( (int) ((uchar(d)<<24) | (uchar(c)<<16) | (uchar(b)<<8) | uchar(a)) )
#define DIM(arr) (sizeof(arr)/sizeof(arr[0]))

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

    MjpegWriter::MjpegWriter() : isOpen(false), outFile(0), outformat(1), outfps(20),
        FrameNum(0), quality(80), NumOfChunks(10) {}

    int MjpegWriter::Open(char* outfile, uchar fps, Size ImSize)
    {
        tencoding = 0;
        if (isOpen) return -4;
        if (fps < 1) return -3;
        if (!(outFile = fopen(outfile, "wb+")))
            return -1;
        outfps = fps;
        width = ImSize.width;
        height = ImSize.height;

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
        if ((pBufSize = toJPGframe(Im.data, width, height, 12, pBuf)) < 0)
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

    int MjpegWriter::toJPGframe(const uchar * data, uint width, uint height, int step, void *& pBuf)
    {
        const int req_comps = 3; // request BGR image, if (BGRA) req_comps = 4; 
        params param;
        param.m_quality = quality;
        param.m_subsampling = H2V2;
        int buf_size = width * height * 3; // allocate a buffer that's hopefully big enough (this is way overkill for jpeg)
        if (buf_size < 1024) buf_size = 1024;
        pBuf = malloc(buf_size);
        jpeg_encoder dst_image;
        if (!dst_image.compress_image_to_jpeg_file_in_memory(pBuf, buf_size, width, height, req_comps, data, param))
            return -1;
        return buf_size;
    }

#define JPGE_MAX(a,b) (((a)>(b))?(a):(b))
#define JPGE_MIN(a,b) (((a)<(b))?(a):(b))

        static inline void *jpge_malloc(size_t nSize) { return malloc(nSize); }
        static inline void jpge_free(void *p) { free(p); }

        // Various JPEG enums and tables.
        enum { M_SOF0 = 0xC0, M_DHT = 0xC4, M_SOI = 0xD8, M_EOI = 0xD9, M_SOS = 0xDA, M_DQT = 0xDB, M_APP0 = 0xE0 };
        enum { DC_LUM_CODES = 12, AC_LUM_CODES = 256, DC_CHROMA_CODES = 12, AC_CHROMA_CODES = 256, MAX_HUFF_SYMBOLS = 257, MAX_HUFF_CODESIZE = 32 };

        static uchar s_zag[64] = { 0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };
        static short s_std_lum_quant[64] = { 16, 11, 12, 14, 12, 10, 16, 14, 13, 14, 18, 17, 16, 19, 24, 40, 26, 24, 22, 22, 24, 49, 35, 37, 29, 40, 58, 51, 61, 60, 57, 51, 56, 55, 64, 72, 92, 78, 64, 68, 87, 69, 55, 56, 80, 109, 81, 87, 95, 98, 103, 104, 103, 62, 77, 113, 121, 112, 100, 120, 92, 101, 103, 99 };
        static short s_std_croma_quant[64] = { 17, 18, 18, 24, 21, 24, 47, 26, 26, 47, 99, 66, 56, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99 };
        static uchar s_dc_lum_bits[17] = { 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
        static uchar s_dc_lum_val[DC_LUM_CODES] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        static uchar s_ac_lum_bits[17] = { 0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
        static uchar s_ac_lum_val[AC_LUM_CODES] =
        {
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
            0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
            0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
            0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
            0xf9, 0xfa
        };
        static uchar s_dc_chroma_bits[17] = { 0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
        static uchar s_dc_chroma_val[DC_CHROMA_CODES] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        static uchar s_ac_chroma_bits[17] = { 0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
        static uchar s_ac_chroma_val[AC_CHROMA_CODES] =
        {
            0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
            0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
            0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
            0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
            0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
            0xf9, 0xfa
        };

        // Low-level helper functions.
        template <class T> inline void clear_obj(T &obj) { memset(&obj, 0, sizeof(obj)); }

        const int YR = 19595, YG = 38470, YB = 7471, CB_R = -11059, CB_G = -21709, CB_B = 32768, CR_R = 32768, CR_G = -27439, CR_B = -5329;

        static uchar clamp_table[1024];
        static bool init_clamp_table = false;

        static inline uchar clamp(int i) { if (static_cast<uint>(i) > 255U) { i = clamp_table[(i)+256]; } return static_cast<uchar>(i); }

        static const int BITS = 10, SCALE = 1 << BITS;
        static const float MAX_M = (float)(1 << (15 - BITS));
        static const short m00 = static_cast<short>(-0.081f * SCALE), m01 = static_cast<short>(-0.419f * SCALE),
            m02 = static_cast<short>(0.5f * SCALE), m10 = static_cast<short>(0.5f * SCALE),
            m11 = static_cast<short>(-0.331f * SCALE), m12 = static_cast<short>(-0.169f * SCALE),
            m20 = static_cast<short>(0.114f * SCALE), m21 = static_cast<short>(0.587f  * SCALE),
            m22 = static_cast<short>(0.299f * SCALE);
        static int m03 = static_cast<int>((128 + 0.5f) * SCALE), m13 = static_cast<int>((0.5f + 128) * SCALE),
            m23 = static_cast<int>(0.5f * SCALE);

        static void BGR_to_YCC(uchar *pDstY, uchar *pDstCb, uchar *pDstCr, const uchar *pSrc, int num_pixels)
        {
            __m128i m0 = _mm_setr_epi16(0, m00, m01, m02, m00, m01, m02, 0);
            __m128i m1 = _mm_setr_epi16(0, m10, m11, m12, m10, m11, m12, 0);
            __m128i m2 = _mm_setr_epi16(0, m20, m21, m22, m20, m21, m22, 0);
            __m128i m3 = _mm_setr_epi32(m03, m13, m23, 0);
            __m128i z = _mm_setzero_si128(), t0, t1, t2, r0, r1, v0, v1, v2, v3;
            int x = 0;

#if SSE
            for (; x <= (num_pixels - 8) * 3; x += 8 * 3, pDstCr += 8, pDstCb += 8, pDstY += 8)
            {
                v0 = _mm_loadl_epi64((const __m128i*)(pSrc + x));
                v1 = _mm_loadl_epi64((const __m128i*)(pSrc + x + 8));
                v2 = _mm_loadl_epi64((const __m128i*)(pSrc + x + 16)), v3;
                v0 = _mm_unpacklo_epi8(v0, z); // b0 g0 r0 b1 g1 r1 b2 g2
                v1 = _mm_unpacklo_epi8(v1, z); // r2 b3 g3 r3 b4 g4 r4 b5
                v2 = _mm_unpacklo_epi8(v2, z); // g5 r5 b6 g6 r6 b7 g7 r7

                v3 = _mm_srli_si128(v2, 2); // ? b6 g6 r6 b7 g7 r7 0
                v2 = _mm_or_si128(_mm_slli_si128(v2, 10), _mm_srli_si128(v1, 6)); // ? b4 g4 r4 b5 g5 r5 ?
                v1 = _mm_or_si128(_mm_slli_si128(v1, 6), _mm_srli_si128(v0, 10)); // ? b2 g2 r2 b3 g3 r3 ?
                v0 = _mm_slli_si128(v0, 2); // 0 b0 g0 r0 b1 g1 r1 ?

                // process pixels 0 & 1
                t0 = _mm_madd_epi16(v0, m0); // a0 b0 a1 b1
                t1 = _mm_madd_epi16(v0, m1); // c0 d0 c1 d1
                t2 = _mm_madd_epi16(v0, m2); // e0 f0 e1 f1
                v0 = _mm_unpacklo_epi32(t0, t1); // a0 c0 b0 d0
                t0 = _mm_unpackhi_epi32(t0, t1); // a1 c1 b1 d1
                t1 = _mm_unpacklo_epi32(t2, z);  // e0 0 f0 0
                t2 = _mm_unpackhi_epi32(t2, z);  // e1 0 f1 0
                r0 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(v0, t1), _mm_unpackhi_epi64(v0, t1)), m3); // V0 U0 Y0 0
                r1 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(t0, t2), _mm_unpackhi_epi64(t0, t2)), m3); // V1 U1 Y1 0
                r0 = _mm_srai_epi32(r0, BITS);
                r1 = _mm_srai_epi32(r1, BITS);
                v0 = _mm_packs_epi32(r0, r1); // V0 U0 Y0 0 V1 U1 Y1 0

                // process pixels 2 & 3
                t0 = _mm_madd_epi16(v1, m0); // a0 b0 a1 b1
                t1 = _mm_madd_epi16(v1, m1); // c0 d0 c1 d1
                t2 = _mm_madd_epi16(v1, m2); // e0 f0 e1 f1
                v1 = _mm_unpacklo_epi32(t0, t1); // a0 c0 b0 d0
                t0 = _mm_unpackhi_epi32(t0, t1); // a1 b1 c1 d1
                t1 = _mm_unpacklo_epi32(t2, z);  // e0 0 f0 0
                t2 = _mm_unpackhi_epi32(t2, z);  // e1 0 f1 0
                r0 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(v1, t1), _mm_unpackhi_epi64(v1, t1)), m3); // V2 U2 Y2 0
                r1 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(t0, t2), _mm_unpackhi_epi64(t0, t2)), m3); // V3 U3 Y3 0
                r0 = _mm_srai_epi32(r0, BITS);
                r1 = _mm_srai_epi32(r1, BITS);
                v1 = _mm_packs_epi32(r0, r1); // V2 U2 Y2 0 V3 U3 Y3 0

                r0 = _mm_unpacklo_epi16(v0, v1); // V0 V2 U0 U2 Y0 Y2 0 0
                v1 = _mm_unpackhi_epi16(v0, v1); // V1 V3 U1 U3 Y1 Y3 0 0
                v0 = _mm_unpacklo_epi16(r0, v1); // V0 V1 V2 V3 U0 U1 U2 U3
                v1 = _mm_unpackhi_epi16(r0, v1); // Y0 Y1 Y2 Y3 0 0 0 0

                //  process pixels 4 & 5
                t0 = _mm_madd_epi16(v2, m0); // a0 b0 a1 b1
                t1 = _mm_madd_epi16(v2, m1); // c0 d0 c1 d1
                t2 = _mm_madd_epi16(v2, m2); // e0 f0 e1 f1
                v2 = _mm_unpacklo_epi32(t0, t1); // a0 c0 b0 d0
                t0 = _mm_unpackhi_epi32(t0, t1); // a1 b1 c1 d1
                t1 = _mm_unpacklo_epi32(t2, z);  // e0 0 f0 0
                t2 = _mm_unpackhi_epi32(t2, z);  // e1 0 f1 0
                r0 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(v2, t1), _mm_unpackhi_epi64(v2, t1)), m3); // V4 U4 Y4 0
                r1 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(t0, t2), _mm_unpackhi_epi64(t0, t2)), m3); // V5 U5 Y5 0
                r0 = _mm_srai_epi32(r0, BITS);
                r1 = _mm_srai_epi32(r1, BITS);
                v2 = _mm_packs_epi32(r0, r1); // V4 U4 Y4 0 V5 U5 Y5 0

                // proce ss pixels 6 & 7
                t0 = _mm_madd_epi16(v3, m0); // a0 b0 a1 b1
                t1 = _mm_madd_epi16(v3, m1); // c0 d0 c1 d1
                t2 = _mm_madd_epi16(v3, m2); // e0 f0 e1 f1
                v3 = _mm_unpacklo_epi32(t0, t1); // a0 c0 b0 d0
                t0 = _mm_unpackhi_epi32(t0, t1); // a1 b1 c1 d1
                t1 = _mm_unpacklo_epi32(t2, z);  // e0 0 f0 0
                t2 = _mm_unpackhi_epi32(t2, z);  // e1 0 f1 0
                r0 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(v3, t1), _mm_unpackhi_epi64(v3, t1)), m3); // V6 U6 Y6 0
                r1 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(t0, t2), _mm_unpackhi_epi64(t0, t2)), m3); // V7 U7 Y7 0
                r0 = _mm_srai_epi32(r0, BITS);
                r1 = _mm_srai_epi32(r1, BITS);
                v3= _mm_packs_epi32(r0, r1); // V6 U6 Y6 0 V7 U7 Y7 0

                r0 = _mm_unpacklo_epi16(v2, v3); // V4 V6 U4 U6 Y4 Y6 0 0
                v3 = _mm_unpackhi_epi16(v2, v3); // V5 V7 U5 U7 Y5 Y7 0 0
                v2 = _mm_unpacklo_epi16(r0, v3); // V4 V5 V6 V7 U4 U5 U6 U7
                v3 = _mm_unpackhi_epi16(r0, v3); // Y4 Y5 Y6 Y7 0 0 0 0

                t0 = _mm_unpacklo_epi64(v0, v2); // V0 ... V7
                t1 = _mm_unpackhi_epi64(v0, v2); // U0 ... U7
                t2 = _mm_unpacklo_epi64(v1, v3); // Y0 ... Y7

                _mm_storel_epi64((__m128i*)pDstCr, _mm_packus_epi16(t0, t0));
                _mm_storel_epi64((__m128i*)pDstCb, _mm_packus_epi16(t1, t1));
                _mm_storel_epi64((__m128i*)pDstY, _mm_packus_epi16(t2, t2));
            }
#endif
            for (; x < num_pixels * 3; x += 3)
            {
                int v0 = pSrc[x], v1 = pSrc[x + 1], v2 = pSrc[x + 2];
                uchar t0 = clamp((m00*v0 + m01*v1 + m02*v2 + m03) >> BITS);
                uchar t1 = clamp((m10*v0 + m11*v1 + m12*v2 + m13) >> BITS);
                uchar t2 = static_cast<uchar>((m20*v0 + m21*v1 + m22*v2 + m23) >> BITS);
                *pDstY++ = t2; *pDstCb++ = t1; *pDstCr++ = t0;
            }
            return;
        }


        static void RGBA_to_YCC(uchar* pDst, const uchar *pSrc, int num_pixels)
        {
            for (; num_pixels; pDst += 3, pSrc += 4, num_pixels--)
            {
                const int r = pSrc[0], g = pSrc[1], b = pSrc[2];
                pDst[0] = static_cast<uchar>((r * YR + g * YG + b * YB + 32768) >> 16);
                pDst[1] = clamp(128 + ((r * CB_R + g * CB_G + b * CB_B + 32768) >> 16));
                pDst[2] = clamp(128 + ((r * CR_R + g * CR_G + b * CR_B + 32768) >> 16));
            }
        }

        // Forward DCT - DCT derived from jfdctint.
        enum { CONST_BITS = 13, ROW_BITS = 2 };
#define DCT_DESCALE(x, n) (((x) + (((int)1) << ((n) - 1))) >> (n)) 
#define DCT_MUL(var, c) (static_cast<short>(var) * static_cast<int>(c))
#define DCT1D(s0, s1, s2, s3, s4, s5, s6, s7) \
    int t0 = s0 + s7, t7 = s0 - s7, t1 = s1 + s6, t6 = s1 - s6, t2 = s2 + s5, t5 = s2 - s5, t3 = s3 + s4, t4 = s3 - s4; \
    int t10 = t0 + t3, t13 = t0 - t3, t11 = t1 + t2, t12 = t1 - t2; \
    int u1 = DCT_MUL(t12 + t13, 4433); \
    s2 = u1 + DCT_MUL(t13, 6270); \
    s6 = u1 + DCT_MUL(t12, -15137); \
    u1 = t4 + t7; \
    int u2 = t5 + t6, u3 = t4 + t6, u4 = t5 + t7; \
    int z5 = DCT_MUL(u3 + u4, 9633); \
    t4 = DCT_MUL(t4, 2446); t5 = DCT_MUL(t5, 16819); \
    t6 = DCT_MUL(t6, 25172); t7 = DCT_MUL(t7, 12299); \
    u1 = DCT_MUL(u1, -7373); u2 = DCT_MUL(u2, -20995); \
    u3 = DCT_MUL(u3, -16069); u4 = DCT_MUL(u4, -3196); \
    u3 += z5; u4 += z5; \
    s0 = t10 + t11; s1 = t7 + u1 + u4; s3 = t6 + u2 + u3; s4 = t10 - t11; s5 = t5 + u2 + u4; s7 = t4 + u1 + u3;

        void jpeg_encoder::DCT2D(int comp)
        {
            int c, shift = 128;
            int *q = m_sample_array;
            uchar *q_uchar = m_sample_array_uchar;
            for (c = 7; c >= 0; c--, q += 8, q_uchar += 8)
            {
                int s0 = (int)q_uchar[0] - shift, s1 = (int)q_uchar[1] - shift, s2 = (int)q_uchar[2] - shift, s3 = (int)q_uchar[3] - shift,
                    s4 = (int)q_uchar[4] - shift, s5 = (int)q_uchar[5] - shift, s6 = (int)q_uchar[6] - shift, s7 = (int)q_uchar[7] - shift;
                DCT1D(s0, s1, s2, s3, s4, s5, s6, s7);
                q[0] = s0 << ROW_BITS; q[1] = DCT_DESCALE(s1, CONST_BITS - ROW_BITS); q[2] = DCT_DESCALE(s2, CONST_BITS - ROW_BITS); q[3] = DCT_DESCALE(s3, CONST_BITS - ROW_BITS);
                q[4] = s4 << ROW_BITS; q[5] = DCT_DESCALE(s5, CONST_BITS - ROW_BITS); q[6] = DCT_DESCALE(s6, CONST_BITS - ROW_BITS); q[7] = DCT_DESCALE(s7, CONST_BITS - ROW_BITS);
            }
            for (q = m_sample_array, q_uchar = m_sample_array_uchar, c = 7; c >= 0; c--, q++, q_uchar++)
            {
                int s0 = q[0 * 8], s1 = q[1 * 8], s2 = q[2 * 8], s3 = q[3 * 8], s4 = q[4 * 8], s5 = q[5 * 8], s6 = q[6 * 8], s7 = q[7 * 8];
                DCT1D(s0, s1, s2, s3, s4, s5, s6, s7);
                q[0 * 8] = DCT_DESCALE(s0, ROW_BITS + 3); q[1 * 8] = DCT_DESCALE(s1, CONST_BITS + ROW_BITS + 3);
                q[2 * 8] = DCT_DESCALE(s2, CONST_BITS + ROW_BITS + 3); q[3 * 8] = DCT_DESCALE(s3, CONST_BITS + ROW_BITS + 3);
                q[4 * 8] = DCT_DESCALE(s4, ROW_BITS + 3); q[5 * 8] = DCT_DESCALE(s5, CONST_BITS + ROW_BITS + 3); 
                q[6 * 8] = DCT_DESCALE(s6, CONST_BITS + ROW_BITS + 3); q[7 * 8] = DCT_DESCALE(s7, CONST_BITS + ROW_BITS + 3);
            }
        }

        struct sym_freq { uint m_key, m_sym_index; };

        // JPEG marker generation.
        void jpeg_encoder::emit_byte(uchar i)
        {
            m_all_stream_writes_succeeded = m_all_stream_writes_succeeded && m_pStream->put_obj(i);
        }

        void jpeg_encoder::emit_word(uint i)
        {
            emit_byte(uchar(i >> 8)); emit_byte(uchar(i & 0xFF));
        }

        void jpeg_encoder::emit_marker(int marker)
        {
            emit_byte(uchar(0xFF)); emit_byte(uchar(marker));
        }

        // Emit JFIF marker
        void jpeg_encoder::emit_jfif_app0()
        {
            emit_marker(M_APP0);
            emit_word(2 + 4 + 1 + 2 + 1 + 2 + 2 + 1 + 1);
            emit_byte(0x4A); emit_byte(0x46); emit_byte(0x49); emit_byte(0x46); /* Identifier: ASCII "JFIF" */
            emit_byte(0);
            emit_byte(1);      /* Major version */
            emit_byte(1);      /* Minor version */
            emit_byte(0);      /* Density unit */
            emit_word(1);
            emit_word(1);
            emit_byte(0);      /* No thumbnail image */
            emit_byte(0);
        }

        // Emit quantization tables
        void jpeg_encoder::emit_dqt()
        {
            for (int i = 0; i < ((m_num_components == 3) ? 2 : 1); i++)
            {
                emit_marker(M_DQT);
                emit_word(64 + 1 + 2);
                emit_byte(static_cast<uchar>(i));
                for (int j = 0; j < 64; j++)
                    emit_byte(static_cast<uchar>(m_quantization_tables[i][j]));
            }
        }

        // Emit start of frame marker
        void jpeg_encoder::emit_sof()
        {
            emit_marker(M_SOF0);                           /* baseline */
            emit_word(3 * m_num_components + 2 + 5 + 1);
            emit_byte(8);                                  /* precision */
            emit_word(m_image_y);
            emit_word(m_image_x);
            emit_byte(m_num_components);
            for (int i = 0; i < m_num_components; i++)
            {
                emit_byte(static_cast<uchar>(i + 1));                                   /* component ID     */
                emit_byte((m_comp_h_samp[i] << 4) + m_comp_v_samp[i]);  /* h and v sampling */
                emit_byte(i > 0);                                   /* quant. table num */
            }
        }

        // Emit Huffman table.
        void jpeg_encoder::emit_dht(uchar *bits, uchar *val, int index, bool ac_flag)
        {
            emit_marker(M_DHT);

            int length = 0;
            for (int i = 1; i <= 16; i++)
                length += bits[i];

            emit_word(length + 2 + 1 + 16);
            emit_byte(static_cast<uchar>(index + (ac_flag << 4)));

            for (int i = 1; i <= 16; i++)
                emit_byte(bits[i]);

            for (int i = 0; i < length; i++)
                emit_byte(val[i]);
        }

        // Emit all Huffman tables.
        void jpeg_encoder::emit_dhts()
        {
            emit_dht(m_huff_bits[0 + 0], m_huff_val[0 + 0], 0, false);
            emit_dht(m_huff_bits[2 + 0], m_huff_val[2 + 0], 0, true);
            if (m_num_components == 3)
            {
                emit_dht(m_huff_bits[0 + 1], m_huff_val[0 + 1], 1, false);
                emit_dht(m_huff_bits[2 + 1], m_huff_val[2 + 1], 1, true);
            }
        }

        // emit start of scan
        void jpeg_encoder::emit_sos()
        {
            emit_marker(M_SOS);
            emit_word(2 * m_num_components + 2 + 1 + 3);
            emit_byte(m_num_components);
            for (int i = 0; i < m_num_components; i++)
            {
                emit_byte(static_cast<uchar>(i + 1));
                if (i == 0)
                    emit_byte((0 << 4) + 0);
                else
                    emit_byte((1 << 4) + 1);
            }
            emit_byte(0);     /* spectral selection */
            emit_byte(63);
            emit_byte(0);
        }

        // Emit all markers at beginning of image file.
        void jpeg_encoder::emit_markers()
        {
            emit_marker(M_SOI);
            emit_jfif_app0();
            emit_dqt();
            emit_sof();
            emit_dhts();
            emit_sos();
        }

        // Compute the actual canonical Huffman codes/code sizes given the JPEG huff bits and val arrays.
        void jpeg_encoder::compute_huffman_table(uint *codes, uchar *code_sizes, uchar *bits, uchar *val)
        {
            int i, l, last_p, si;
            uchar huff_size[257];
            uint huff_code[257];
            uint code;

            int p = 0;
            for (l = 1; l <= 16; l++)
            for (i = 1; i <= bits[l]; i++)
                huff_size[p++] = (char)l;

            huff_size[p] = 0; last_p = p; // write sentinel

            code = 0; si = huff_size[0]; p = 0;

            while (huff_size[p])
            {
                while (huff_size[p] == si)
                    huff_code[p++] = code++;
                code <<= 1;
                si++;
            }

            memset(codes, 0, sizeof(codes[0]) * 256);
            memset(code_sizes, 0, sizeof(code_sizes[0]) * 256);
            for (p = 0; p < last_p; p++)
            {
                codes[val[p]] = huff_code[p];
                code_sizes[val[p]] = huff_size[p];
            }
        }

        // Quantization table generation.
        void jpeg_encoder::compute_quant_table(int *pDst, short *pSrc)
        {
            int q;
            if (m_params.m_quality < 50)
                q = 5000 / m_params.m_quality;
            else
                q = 200 - m_params.m_quality * 2;
            for (int i = 0; i < 64; i++)
            {
                int j = *pSrc++; j = (j * q + 50L) / 100L;
                *pDst++ = JPGE_MIN(JPGE_MAX(j, 1), 255);
            }
        }

        // Higher-level methods.
        void jpeg_encoder::first_pass_init()
        {
            m_bit_buffer = 0; m_bits_in = 0;
            memset(m_last_dc_val, 0, 3 * sizeof(m_last_dc_val[0]));
            m_mcu_y_ofs = 0;
            m_pass_num = 1;
        }

        bool jpeg_encoder::second_pass_init()
        {
            compute_huffman_table(&m_huff_codes[0 + 0][0], &m_huff_code_sizes[0 + 0][0], m_huff_bits[0 + 0], m_huff_val[0 + 0]);
            compute_huffman_table(&m_huff_codes[2 + 0][0], &m_huff_code_sizes[2 + 0][0], m_huff_bits[2 + 0], m_huff_val[2 + 0]);
            if (m_num_components > 1)
            {
                compute_huffman_table(&m_huff_codes[0 + 1][0], &m_huff_code_sizes[0 + 1][0], m_huff_bits[0 + 1], m_huff_val[0 + 1]);
                compute_huffman_table(&m_huff_codes[2 + 1][0], &m_huff_code_sizes[2 + 1][0], m_huff_bits[2 + 1], m_huff_val[2 + 1]);
            }
            first_pass_init();
            emit_markers();
            m_pass_num = 2;
            return true;
        }

        bool jpeg_encoder::jpg_open(int p_x_res, int p_y_res, int src_channels)
        {
            m_num_components = 3;
            m_comp_h_samp[0] = 2; m_comp_v_samp[0] = 2;
            m_comp_h_samp[1] = 1; m_comp_v_samp[1] = 1;
            m_comp_h_samp[2] = 1; m_comp_v_samp[2] = 1;
            m_mcu_x = 16; m_mcu_y = 16;

            m_image_x = p_x_res; m_image_y = p_y_res;
            m_image_bpp = src_channels;
            m_image_bpl = m_image_x * src_channels;
            m_image_x_mcu = (m_image_x + m_mcu_x - 1) & (~(m_mcu_x - 1));
            m_image_y_mcu = (m_image_y + m_mcu_y - 1) & (~(m_mcu_y - 1));
            m_image_bpl_xlt = m_image_x * m_num_components;
            m_image_bpl_mcu = m_image_x_mcu * m_num_components;
            m_mcus_per_row = m_image_x_mcu / m_mcu_x;

            if (((m_mcu_linesY[0] = static_cast<uchar*>(jpge_malloc(m_image_x_mcu * m_mcu_y))) == 0) || 
                ((m_mcu_linesCb[0] = static_cast<uchar*>(jpge_malloc(m_image_x_mcu * m_mcu_y))) == 0) ||
                ((m_mcu_linesCr[0] = static_cast<uchar*>(jpge_malloc(m_image_x_mcu * m_mcu_y))) == 0)) return false;
            for (int i = 1; i < m_mcu_y; i++)
            {
                m_mcu_linesY[i] = m_mcu_linesY[i - 1] + m_image_x_mcu;
                m_mcu_linesCb[i] = m_mcu_linesCb[i - 1] + m_image_x_mcu;
                m_mcu_linesCr[i] = m_mcu_linesCr[i - 1] + m_image_x_mcu;
            }

            compute_quant_table(m_quantization_tables[0], s_std_lum_quant);
            compute_quant_table(m_quantization_tables[1], m_params.m_no_chroma_discrim_flag ? s_std_lum_quant : s_std_croma_quant);

            m_out_buf_left = JPGE_OUT_BUF_SIZE;
            m_pOut_buf = m_out_buf;

            if (m_params.m_two_pass_flag)
            {
                clear_obj(m_huff_count);
                first_pass_init();
            }
            else
            {
                memcpy(m_huff_bits[0 + 0], s_dc_lum_bits, 17);    memcpy(m_huff_val[0 + 0], s_dc_lum_val, DC_LUM_CODES);
                memcpy(m_huff_bits[2 + 0], s_ac_lum_bits, 17);    memcpy(m_huff_val[2 + 0], s_ac_lum_val, AC_LUM_CODES);
                memcpy(m_huff_bits[0 + 1], s_dc_chroma_bits, 17); memcpy(m_huff_val[0 + 1], s_dc_chroma_val, DC_CHROMA_CODES);
                memcpy(m_huff_bits[2 + 1], s_ac_chroma_bits, 17); memcpy(m_huff_val[2 + 1], s_ac_chroma_val, AC_CHROMA_CODES);
                if (!second_pass_init()) return false;   // in effect, skip over the first pass
            }
            return m_all_stream_writes_succeeded;
        }
        void jpeg_encoder::load_block_8_8(int x, int y)
        {
#if SSE
            uchar *pDst = m_sample_array_uchar;
            x <<= 3;
            y <<= 3;
            __m128i str;
            for (int i = 0; i < 8; i++, pDst += 8)
            {
                str = _mm_loadl_epi64((const __m128i*)(m_mcu_linesY[y + i] + x));
                _mm_storel_epi64((__m128i*)pDst, str);
            }


#else
            uchar *pDst = m_sample_array_uchar;
            x <<= 3;
            y <<= 3;
            const int n_bytes = 8;
            for (int i = 0; i < 8; i++, pDst += 8)
                memcpy(pDst, m_mcu_linesY[y + i] + x, n_bytes);
#endif
        }
        void jpeg_encoder::load_block_16_8(int x, int comp)
        {
            uchar *pSrc1, *pSrc2, **pSrc;
            x <<= 4;
            int a = 0, b = 2;
            if (comp == 1)
                pSrc = m_mcu_linesCb;
            else
                pSrc = m_mcu_linesCr;
#if SSE
            uchar *pDst = m_sample_array_uchar;
            __m128i r0, r1, a0, b0, a1, b1, res0, res1; 
            __m128i z = _mm_setzero_si128(), mask = _mm_set1_epi16(255), delta = _mm_set1_epi16(2);

            const int di = 4;
            for (int i = 0; i < 16; i += di, pDst += di * 4)
            {
                pSrc1 = pSrc[i + 0] + x;
                pSrc2 = pSrc[i + 1] + x;
                r0 = _mm_loadu_si128((const __m128i*)pSrc1);
                r1 = _mm_loadu_si128((const __m128i*)pSrc2);
                a0 = _mm_and_si128(r0, mask); // u0 0 u2 0 u4 0 ...
                b0 = _mm_srli_epi16(r0, 8);   // u1 0 u3 0 u5 0 ...
                a1 = _mm_and_si128(r1, mask); // u0' 0 u2' 0 u4' 0 ...
                b1 = _mm_srli_epi16(r1, 8);   // u1' 0 u3' 0 u5' 0 ...
                res0 = _mm_add_epi16(_mm_add_epi16(a0, b0), _mm_add_epi16(a1, b1));
                res0 = _mm_srli_epi16(_mm_add_epi16(res0, delta), 2);

                pSrc1 = pSrc[i + 2] + x;
                pSrc2 = pSrc[i + 3] + x;
                r0 = _mm_loadu_si128((const __m128i*)pSrc1);
                r1 = _mm_loadu_si128((const __m128i*)pSrc2);
                a0 = _mm_and_si128(r0, mask); // u0 0 u2 0 u4 0 ...
                b0 = _mm_srli_epi16(r0, 8);   // u1 0 u3 0 u5 0 ...
                a1 = _mm_and_si128(r1, mask); // u0' 0 u2' 0 u4' 0 ...
                b1 = _mm_srli_epi16(r1, 8);   // u1' 0 u3' 0 u5' 0 ...
                res1 = _mm_add_epi16(_mm_add_epi16(a0, b0), _mm_add_epi16(a1, b1));
                res1 = _mm_srli_epi16(_mm_add_epi16(res1, delta), 2);
                
                _mm_storeu_si128((__m128i*)pDst, _mm_packus_epi16(res0, res1));
            }
#else
            uchar *pDst = m_sample_array_uchar;
            for (int i = 0; i < 16; i += 2, pDst += 8)
            {
                pSrc1 = pSrc[i + 0] + x;
                pSrc2 = pSrc[i + 1] + x;
                pDst[0] = (uchar)((pSrc1[0] + pSrc1[1] + pSrc2[0] + pSrc2[1] + a) >> 2); pDst[1] = (uchar)((pSrc1[2] + pSrc1[3] + pSrc2[2] + pSrc2[3] + b) >> 2);
                pDst[2] = (uchar)((pSrc1[4] + pSrc1[5] + pSrc2[4] + pSrc2[5] + a) >> 2); pDst[3] = (uchar)((pSrc1[6] + pSrc1[7] + pSrc2[6] + pSrc2[7] + b) >> 2);
                pDst[4] = (uchar)((pSrc1[8] + pSrc1[9] + pSrc2[8] + pSrc2[9] + a) >> 2); pDst[5] = (uchar)((pSrc1[10] + pSrc1[11] + pSrc2[10] + pSrc2[11] + b) >> 2);
                pDst[6] = (uchar)((pSrc1[12] + pSrc1[13] + pSrc2[12] + pSrc2[13] + a) >> 2); pDst[7] = (uchar)((pSrc1[14] + pSrc1[15] + pSrc2[14] + pSrc2[15] + b) >> 2);
                int temp = a; a = b; b = temp;
            }
#endif
        }

        void jpeg_encoder::load_quantized_coefficients(int component_num)
        {
            int *q = m_quantization_tables[component_num > 0];
            short *pDst = m_coefficient_array;
            for (int i = 0; i < 64; i++)
            {
                sample_array_t j = m_sample_array[s_zag[i]];
                if (j < 0)
                {
                    if ((j = -j + (*q >> 1)) < *q)
                        *pDst++ = 0;
                    else
                        *pDst++ = static_cast<short>(-(j / *q));
                }
                else
                {
                    if ((j = j + (*q >> 1)) < *q)
                        *pDst++ = 0;
                    else
                        *pDst++ = static_cast<short>((j / *q));
                }
                q++;
            }
        }

        void jpeg_encoder::flush_output_buffer()
        {
            if (m_out_buf_left != JPGE_OUT_BUF_SIZE)
                m_all_stream_writes_succeeded = m_all_stream_writes_succeeded && m_pStream->put_buf(m_out_buf, JPGE_OUT_BUF_SIZE - m_out_buf_left);
            m_pOut_buf = m_out_buf;
            m_out_buf_left = JPGE_OUT_BUF_SIZE;
        }

        void jpeg_encoder::put_bits(uint bits, uint len)
        {
            m_bit_buffer |= ((uint)bits << (24 - (m_bits_in += len)));
            while (m_bits_in >= 8)
            {
                uchar c;
#define JPGE_PUT_BYTE(c) { *m_pOut_buf++ = (c); if (--m_out_buf_left == 0) flush_output_buffer(); }
                JPGE_PUT_BYTE(c = (uchar)((m_bit_buffer >> 16) & 0xFF));
                if (c == 0xFF) JPGE_PUT_BYTE(0);
                m_bit_buffer <<= 8;
                m_bits_in -= 8;
            }
        }

        void jpeg_encoder::code_coefficients_pass_two(int component_num)
        {
            int i, j, run_len, nbits, temp1, temp2;
            short *pSrc = m_coefficient_array;
            uint *codes[2];
            uchar *code_sizes[2];

            if (component_num == 0)
            {
                codes[0] = m_huff_codes[0 + 0]; codes[1] = m_huff_codes[2 + 0];
                code_sizes[0] = m_huff_code_sizes[0 + 0]; code_sizes[1] = m_huff_code_sizes[2 + 0];
            }
            else
            {
                codes[0] = m_huff_codes[0 + 1]; codes[1] = m_huff_codes[2 + 1];
                code_sizes[0] = m_huff_code_sizes[0 + 1]; code_sizes[1] = m_huff_code_sizes[2 + 1];
            }

            temp1 = temp2 = pSrc[0] - m_last_dc_val[component_num];
            m_last_dc_val[component_num] = pSrc[0];

            if (temp1 < 0)
            {
                temp1 = -temp1; temp2--;
            }

            nbits = 0;
            while (temp1)
            {
                nbits++; temp1 >>= 1;
            }

            put_bits(codes[0][nbits], code_sizes[0][nbits]);
            if (nbits) put_bits(temp2 & ((1 << nbits) - 1), nbits);

            for (run_len = 0, i = 1; i < 64; i++)
            {
                if ((temp1 = m_coefficient_array[i]) == 0)
                    run_len++;
                else
                {
                    while (run_len >= 16)
                    {
                        put_bits(codes[1][0xF0], code_sizes[1][0xF0]);
                        run_len -= 16;
                    }
                    if ((temp2 = temp1) < 0)
                    {
                        temp1 = -temp1;
                        temp2--;
                    }
                    nbits = 1;
                    while (temp1 >>= 1)
                        nbits++;
                    j = (run_len << 4) + nbits;
                    put_bits(codes[1][j], code_sizes[1][j]);
                    put_bits(temp2 & ((1 << nbits) - 1), nbits);
                    run_len = 0;
                }
            }
            if (run_len)
                put_bits(codes[1][0], code_sizes[1][0]);
        }

        void jpeg_encoder::code_block(int component_num)
        {
            DCT2D(component_num);
            load_quantized_coefficients(component_num);
            code_coefficients_pass_two(component_num);
        }

        void jpeg_encoder::process_mcu_row()
        {
            for (int i = 0; i < m_mcus_per_row; i++)
            {
                load_block_8_8(i * 2 + 0, 0); code_block(0); load_block_8_8(i * 2 + 1, 0); code_block(0);
                load_block_8_8(i * 2 + 0, 1); code_block(0); load_block_8_8(i * 2 + 1, 1); code_block(0);
                load_block_16_8(i, 1); code_block(1); load_block_16_8(i, 2); code_block(2);
            }
        }

        bool jpeg_encoder::terminate_pass_two()
        {
            put_bits(0x7F, 7);
            flush_output_buffer();
            emit_marker(M_EOI);
            m_pass_num++; // purposely bump up m_pass_num, for debugging
            return true;
        }

        bool jpeg_encoder::process_end_of_image()
        {
            if (m_mcu_y_ofs)
            {
                if (m_mcu_y_ofs < 16) // check here just to shut up static analysis
                {
                    for (int i = m_mcu_y_ofs; i < m_mcu_y; i++)
                    {
                        memcpy(m_mcu_linesY[i], m_mcu_linesY[m_mcu_y_ofs - 1], m_image_x_mcu);
                        memcpy(m_mcu_linesCb[i], m_mcu_linesCb[m_mcu_y_ofs - 1], m_image_x_mcu);
                        memcpy(m_mcu_linesCr[i], m_mcu_linesCr[m_mcu_y_ofs - 1], m_image_x_mcu);
                    }
                }
                process_mcu_row();
            }
                return terminate_pass_two();
        }

        void jpeg_encoder::load_mcu(const void *pSrc)
        {
            const uchar* Psrc = reinterpret_cast<const uchar*>(pSrc);

            uchar* pDstY = m_mcu_linesY[m_mcu_y_ofs]; // OK to write up to m_image_bpl_xlt bytes to pDst
            uchar* pDstCb = m_mcu_linesCb[m_mcu_y_ofs];
            uchar* pDstCr = m_mcu_linesCr[m_mcu_y_ofs];

            //if (m_image_bpp == 4)
            //    RGBA_to_YCC(pDst, Psrc, m_image_x);
            //else
            if (m_image_bpp == 3)
                BGR_to_YCC(pDstY, pDstCb, pDstCr, Psrc, m_image_x);

            // Possibly duplicate pixels at end of scanline if not a multiple of 8 or 16
            const uchar y = pDstY[m_image_x], cb = pDstCb[m_image_x], cr = pDstCr[m_image_x];
            uchar *Y = m_mcu_linesY[m_mcu_y_ofs] + m_image_x;
            uchar *Cb = m_mcu_linesY[m_mcu_y_ofs] + m_image_x;
            uchar *Cr = m_mcu_linesY[m_mcu_y_ofs] + m_image_x;
            for (int i = m_image_x; i < m_image_x_mcu; i++)
            {
                *Y++ = y; *Cb++ = cb; *Cr++ = cr;
            }

            if (++m_mcu_y_ofs == m_mcu_y)
            {
                process_mcu_row();
                m_mcu_y_ofs = 0;
            }
        }

        void jpeg_encoder::clear()
        {
            m_mcu_linesY[0] = 0;
            m_mcu_linesCb[0] = 0;
            m_mcu_linesCr[0] = 0;
            m_pass_num = 0;
            m_all_stream_writes_succeeded = true;
        }

        jpeg_encoder::jpeg_encoder()
        {
            clear();
        }

        jpeg_encoder::~jpeg_encoder()
        {
            deinit();
        }

        bool jpeg_encoder::init(output_stream *pStream, int width, int height, int src_channels, const params &comp_params)
        {
            deinit();
            if (((!pStream) || (width < 1) || (height < 1)) || ((src_channels != 1) && (src_channels != 3) && (src_channels != 4)) || (!comp_params.check())) return false;
            m_pStream = pStream;
            m_params = comp_params;
            return jpg_open(width, height, src_channels);
        }

        void jpeg_encoder::deinit()
        {
            jpge_free(m_mcu_linesY[0]);
            jpge_free(m_mcu_linesCb[0]);
            jpge_free(m_mcu_linesCr[0]);
            clear();
        }

        bool jpeg_encoder::process_scanline(const void* pScanline)
        {
            if ((m_pass_num < 1) || (m_pass_num > 2)) return false;
            if (m_all_stream_writes_succeeded)
            {
                if (!pScanline)
                {
                    if (!process_end_of_image()) return false;
                }
                else
                {
                    load_mcu(pScanline);
                }
            }
            return m_all_stream_writes_succeeded;
        }

        // Higher level wrappers/examples (optional).
#include <stdio.h>

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

        bool jpeg_encoder::compress_image_to_jpeg_file_in_memory(void *&pDstBuf, int &buf_size, int width, int height, int num_channels, const uchar *pImage_data, const params &comp_params)
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

            if (!init(&dst_stream, width, height, num_channels, comp_params))
                return false;

            for (uint pass_index = 0; pass_index < get_total_passes(); pass_index++)
            {
                for (int i = 0; i < height; i++)
                {
                    const uchar* pScanline = pImage_data + i * width * num_channels;
                    if (!process_scanline(pScanline))
                        return false;
                }
                if (!process_scanline(0))
                    return false;
            }
            deinit();

            buf_size = dst_stream.get_size();
            return true;
        }
}

/////////////////////// GrFmtJpegWriter ///////////////////

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

#define fixb         14
#define fix(x, n)    (int)((x)*(1 << (n)) + .5)
#define fix1(x, n)   (x)
#define fixmul(x)    (x)

#define C0_707     fix( 0.707106781f, fixb )
#define C0_924     fix( 0.923879533f, fixb )
#define C0_541     fix( 0.541196100f, fixb )
#define C0_382     fix( 0.382683432f, fixb )
#define C1_306     fix( 1.306562965f, fixb )

#define C1_082     fix( 1.082392200f, fixb )
#define C1_414     fix( 1.414213562f, fixb )
#define C1_847     fix( 1.847759065f, fixb )
#define C2_613     fix( 2.613125930f, fixb )

#define fixc       12
#define b_cb       fix( 1.772, fixc )
#define g_cb      -fix( 0.34414, fixc )
#define g_cr      -fix( 0.71414, fixc )
#define r_cr       fix( 1.402, fixc )

#define y_r        fix( 0.299, fixc )
#define y_g        fix( 0.587, fixc )
#define y_b        fix( 0.114, fixc )

#define cb_r      -fix( 0.1687, fixc )
#define cb_g      -fix( 0.3313, fixc )
#define cb_b       fix( 0.5,    fixc )

#define cr_r       fix( 0.5,    fixc )
#define cr_g      -fix( 0.4187, fixc )
#define cr_b      -fix( 0.0813, fixc )

static const uchar zigzag[] =
{
    0, 8, 1, 2, 9, 16, 24, 17, 10, 3, 4, 11, 18, 25, 32, 40,
    33, 26, 19, 12, 5, 6, 13, 20, 27, 34, 41, 48, 56, 49, 42, 35,
    28, 21, 14, 7, 15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30,
    23, 31, 38, 45, 52, 59, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63,
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

#define postshift 14

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

#if 0
bool  WriteImage(const uchar* data, int step,
    int width, int height, int /*depth*/, int _channels)
{
    assert(data && width > 0 && height > 0);

    if (!m_strm.Open(m_filename)) return false;

    // encode the header and tables
    // for each mcu:
    //   convert rgb to yuv with downsampling (if color).
    //   for every block:
    //     calc dct and quantize
    //     encode block.
    int x, y;
    int i, j;
    const int max_quality = 12;
    int   quality = max_quality;
    WMByteStream& lowstrm = m_strm.m_low_strm;
    int   fdct_qtab[2][64];
    unsigned long huff_dc_tab[2][16];
    unsigned long huff_ac_tab[2][256];
    int  channels = _channels > 1 ? 3 : 1;
    int  x_scale = channels > 1 ? 2 : 1, y_scale = x_scale;
    int  dc_pred[] = { 0, 0, 0 };
    int  x_step = x_scale * 8;
    int  y_step = y_scale * 8;
    int  block[6][64];
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
            int qval = cvRound(qtable[idx] * inv_quality);
            if (qval < 1)
                qval = 1;
            if (qval > 255)
                qval = 255;
            fdct_qtab[i][idx] = cvRound((1 << (postshift + 9)) /
                (qval*chroma_scale*idct_prescale[idx]));
            lowstrm.PutByte(qval);
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
        lowstrm.PutByte((i>0) * 16 + (i>0));// selection of DC & AC tables
    }

    lowstrm.PutWord(0 * 256 + 63);// start and end of spectral selection - for
    // sequental DCT start is 0 and end is 63

    lowstrm.PutByte(0);  // successive approximation bit position 
    // high & low - (0,0) for sequental DCT  

    // encode data
    for (y = 0; y < height; y += y_step, data += y_step*step)
    {
        for (x = 0; x < width; x += x_step)
        {
            int x_limit = x_step;
            int y_limit = y_step;
            const uchar* rgb_data = data + x*_channels;
            int* Y_data = block[0];

            if (x + x_limit > width) x_limit = width - x;
            if (y + y_limit > height) y_limit = height - y;

            memset(block, 0, block_count * 64 * sizeof(block[0][0]));

            if (channels > 1)
            {
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

                    rgb_data -= x_limit*_channels;
                    if (((i + 1) & (y_scale - 1)) == 0)
                    {
                        UV_data += UV_step;
                    }
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
                int* src_ptr = block[i & -2] + (i & 1) * 8;
                const unsigned long* htable = huff_ac_tab[is_chroma];

                aan_fdct8x8(src_ptr, buffer, src_step, fdct_qtab[is_chroma]);

                j = is_chroma + (i > luma_count);
                val = buffer[0] - dc_pred[j];
                dc_pred[j] = buffer[0];

                {
                    float a = (float)val;
                    int cat = (((int&)a >> 23) & 255) - (126 & (val ? -1 : 0));

                    assert(cat <= 11);
                    m_strm.PutHuff(cat, huff_dc_tab[is_chroma]);
                    m_strm.Put(val - (val < 0 ? 1 : 0), cat);
                }

                for (j = 1; j < 64; j++)
                {
                    val = buffer[zigzag[j]];

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
#endif

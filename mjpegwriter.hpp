
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <assert.h>

using namespace cv;
using namespace std;

typedef unsigned char uchar;
typedef unsigned long ulong;

namespace jcodec
{
    enum subsampling_t { Y_ONLY = 0, H1V1 = 1, H2V1 = 2, H2V2 = 3 };

    class output_stream
    {
    public:
        virtual ~output_stream() { };
        virtual bool put_buf(const void* Pbuf, int len) = 0;
        template<class T> inline bool put_obj(const T& obj) { return put_buf(&obj, sizeof(T)); }
    };

    struct params
    {
        inline params() : m_quality(85), m_subsampling((subsampling_t)H2V2), m_no_chroma_discrim_flag(false), m_two_pass_flag(false), block_size(16) { }

        inline bool check() const
        {
            if ((m_quality < 1) || (m_quality > 100)) return false;
            if ((uint)m_subsampling > (uint)H2V2) return false;
            return true;
        }

        int m_quality;
        int block_size;
        subsampling_t m_subsampling;
        bool m_no_chroma_discrim_flag;
        bool m_two_pass_flag;
    };

    class MjpegWriter
    {
    public:
        MjpegWriter();
        int Open(char* outfile, uchar fps, Size ImSize);
        int Write(const Mat &Im);
        int Close();
        bool isOpened();
    private:
        const int NumOfChunks;
        double tencoding;
        FILE *outFile;
        char *outfileName;
        int outformat, outfps, quality;
        int width, height, type, FrameNum;
        int chunkPointer, moviPointer;
        vector<int> FrameOffset, FrameSize, AVIChunkSizeIndex, FrameNumIndexes;
        bool isOpen;

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

    class jpeg_encoder
    {
    public:
        jpeg_encoder();
        ~jpeg_encoder();
        bool init(output_stream *pStream, int width, int height, int src_channels, const params &comp_params = params());
        void deinit();
        bool compress_image_to_jpeg_file_in_memory(void *&pBuf, int &buf_size, int width, int height, int num_channels, const uchar *pImage_data, const params &comp_params = params());

    private:
        jpeg_encoder(const jpeg_encoder &);
        jpeg_encoder &operator =(const jpeg_encoder &);
        typedef int sample_array_t;
        output_stream *m_pStream;
        params m_params;
        bool WriteImage(output_stream *pStream, const uchar* data, int step, int width, int height, int _channels);
        
    };

    // WBaseStream - base class for output streams
    class WBaseStream
    {
    public:
        //methods
        WBaseStream();
        virtual ~WBaseStream();

        virtual bool  Open(output_stream *stream);
        virtual void  Close();
        void          SetBlockSize(int block_size);
        bool          IsOpened();
        int           GetPos();

    protected:

        uchar*  m_start;
        uchar*  m_end;
        uchar*  m_current;
        int     m_block_size;
        int     m_block_pos;
        output_stream*   m_stream;
        bool    m_is_opened;

        virtual void  WriteBlock();
        virtual void  Release();
        virtual void  Allocate();
    };


    // class WLByteStream - uchar-oriented stream.
    // l in prefix means that the least significant uchar of a multi-byte value goes first
    class WLByteStream : public WBaseStream
    {
    public:
        virtual ~WLByteStream();

        void    PutByte(int val);
        void    PutBytes(const void* buffer, int count);
        void    PutWord(int val);
        void    PutDWord(int val);
    };


    // class WLByteStream - uchar-oriented stream.
    // m in prefix means that the least significant uchar of a multi-byte value goes last
    class WMByteStream : public WLByteStream
    {
    public:
        virtual ~WMByteStream();

        void    PutWord(int val);
        void    PutDWord(int val);
    };


    // class WLBitStream - bit-oriented stream.
    // l in prefix means that the least significant bit of a multi-bit value goes first
    class WLBitStream : public WBaseStream
    {
    public:
        virtual ~WLBitStream();

        int     GetPos();
        void    Put(int val, int bits);
        void    PutHuff(int val, const int* table);

    protected:
        int     m_bit_idx;
        int     m_val;
        virtual void  WriteBlock();
    };


    // class WMBitStream - bit-oriented stream.
    // l in prefix means that the least significant bit of a multi-bit value goes first
    class WMBitStream : public WBaseStream
    {
    public:
        WMBitStream();
        virtual ~WMBitStream();

        bool    Open(output_stream *stream);
        void    Close();
        virtual void  Flush();

        int     GetPos();
        void    Put(int val, int bits);
        void    PutHuff(int val, const ulong* table);

    protected:
        int     m_bit_idx;
        ulong   m_pad_val;
        ulong   m_val;
        virtual void  WriteBlock();
        void    ResetBuffer();
    };

    int* bsCreateSourceHuffmanTable(const uchar* src, int* dst,
        int max_bits, int first_bits);
    bool bsCreateDecodeHuffmanTable(const int* src, short* dst, int max_size);
    bool bsCreateEncodeHuffmanTable(const int* src, ulong* dst, int max_size);

    class WJpegBitStream : public WMBitStream
    {
    public:
        WMByteStream  m_low_strm;

        WJpegBitStream();
        ~WJpegBitStream();

        virtual void  Flush();
        virtual bool  Open(output_stream *stream);
        virtual void  Close();

    protected:
        virtual void  WriteBlock();
    };
}
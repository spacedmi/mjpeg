/* СТРУКТУРЫ ЗАГОЛОВКОВ
MainAVIHeader:
DWORD dwMicroSecPerFrame;    //  Количество микросекунд между кадрами для всего файла
DWORD dwMaxBytesPerSec;      //  Примерную максимальную скорость передачи данных файла
DWORD dwReserved1;           // NULL
DWORD dwFlags;               // ??
DWORD dwTotalFrames;         // Общее число кадров, запишется в самом конце
DWORD dwInitialFrames;       // Игнорировать
DWORD dwStreams;             // Число потоков 2: Видео + аудио
DWORD dwSuggestedBufferSize; // Размер буфера для большего chunka'а. Если 0, то буфер задаст программа для воспроизведения
DWORD dwWidth;               // Ширина кадра
DWORD dwHeight;              // Высота кадра
DWORD dwReserved[4];         // NULL, NULL, NULL, NULL
AVIStreamHeader:
FOURCC fccType;              // Может иметь fourcc 'vids','auds', 'txts'...
FOURCC fccHandler;           // cvid 
DWORD dwFlags;               // NULL
DWORD dwPriority;            // NULL
DWORD dwInitialFrames;       // NULL
DWORD dwScale;               // NULL
DWORD dwRate;                // NULL
DWORD dwStart;               // NULL
DWORD dwLength;              // NULL
DWORD dwSuggestedBufferSize; // NULL
DWORD dwQuality;             // NULL
DWORD dwSampleSize;          // NULL
RECT rcFrame;
*/
#include "mjpegwriter.hpp"
#include <fstream>

typedef unsigned long DWORD;
typedef unsigned char BYTE;
typedef DWORD FOURCC;
typedef unsigned short WORD;
#define fourCC(a,b,c,d) ( (DWORD) (((d)<<24) | ((c)<<16) | ((b)<<8) | (a)) )
#define NUM_MICROSEC_PER_SEC 1000000
#define MAX_BYTES_PER_SEC 314572800         // 300 Mb/s
#define STREAMS 1

MjpegWriter::MjpegWriter() : isOpen(false), outFile(NULL), outformat(AVI), outfps(20) { }

int MjpegWriter::Open(const char* outfile, uchar format, uchar fps)
{
    if (fps < 1) return -3;
    switch (format)
    {
        case AVI:
        {
            if (!(outFile = fopen(outfile, "wb+")))
                return -1;
            // Record MainAviHeader. НЕ ЗАБЫТЬ ПОТОМ ЗАПИСАТЬ РАЗМЕР
            DWORD StartAVIRIFF[22] = { fourCC('R', 'I', 'F', 'F'), NULL, fourCC('A', 'V', 'I', ' '),
                fourCC('L', 'I', 'S', 'T'), NULL, fourCC('h', 'd', 'r', 'l'), fourCC('a', 'v', 'i', 'h'),
                sizeof(DWORD) * 14, 1.0f * NUM_MICROSEC_PER_SEC / fps, MAX_BYTES_PER_SEC, NULL,
                NULL, NULL, NULL, STREAMS, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
            fwrite(StartAVIRIFF, sizeof(DWORD), 22, outFile);
            // Record StreamHeader. НЕ ЗАБЫТЬ ПОТОМ ЗАПИСАТЬ РАЗМЕР
            DWORD StreamHEAD[18] = { fourCC('L', 'I', 'S', 'T'), NULL, fourCC('s', 't', 'r', 'l'), fourCC('s', 't', 'r', 'h'),
                sizeof(DWORD)* 13 + sizeof(WORD), fourCC('v', 'i', 'd', 's'), fourCC('c', 'v', 'i', 'd'), NULL, NULL, NULL,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL};
            outfps = fps;
            break;
        }
        default: 
            return -2;
    }
    
    isOpen = true;
    return 1;
}

int MjpegWriter::Close()
{
    // Здесь запись индексов и заполнение структуры AVI
    if (fclose(outFile))
        return -1;
    isOpen = false;
    // Здесь проверка, если в файл не был записан ни один кадр, удалить файл
    return 1;
}

int MjpegWriter::Write(const Mat Im)
{
    if (!isOpen) return -1;
    // Потоковая запись, тут вызывается jpeg энкодер
    return 1;
}

bool MjpegWriter::isOpened()
{
    return isOpen;
}

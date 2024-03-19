#ifndef AVIF_H
#define AVIF_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define CRABBY_AVIF_DEFAULT_IMAGE_SIZE_LIMIT (16384 * 16384)

#define CRABBY_AVIF_DEFAULT_IMAGE_DIMENSION_LIMIT 32768

#define CRABBY_AVIF_DEFAULT_IMAGE_COUNT_LIMIT ((12 * 3600) * 60)

#define CRABBY_AVIF_MAX_AV1_LAYER_COUNT 4

#define CRABBY_AVIF_TRUE 1

#define CRABBY_AVIF_FALSE 0

#define AVIF_STRICT_DISABLED 0

#define AVIF_STRICT_PIXI_REQUIRED (1 << 0)

#define AVIF_STRICT_CLAP_VALID (1 << 1)

#define AVIF_STRICT_ALPHA_ISPE_REQUIRED (1 << 2)

#define AVIF_STRICT_ENABLED ((AVIF_STRICT_PIXI_REQUIRED | AVIF_STRICT_CLAP_VALID) | AVIF_STRICT_ALPHA_ISPE_REQUIRED)

#define CRABBY_AVIF_DIAGNOSTICS_ERROR_BUFFER_SIZE 256

#define CRABBY_AVIF_PLANE_COUNT_YUV 3

#define CRABBY_AVIF_REPETITION_COUNT_INFINITE -1

#define CRABBY_AVIF_REPETITION_COUNT_UNKNOWN -2

#define AVIF_TRANSFORM_NONE 0

#define AVIF_TRANSFORM_PASP (1 << 0)

#define AVIF_TRANSFORM_CLAP (1 << 1)

#define AVIF_TRANSFORM_IROT (1 << 2)

#define AVIF_TRANSFORM_IMIR (1 << 3)

#define AVIF_COLOR_PRIMARIES_BT709 1

#define AVIF_COLOR_PRIMARIES_IEC61966_2_4 1

#define AVIF_COLOR_PRIMARIES_BT2100 9

#define AVIF_COLOR_PRIMARIES_DCI_P3 12

#define AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084 16

typedef enum avifChromaDownsampling {
    AVIF_CHROMA_DOWNSAMPLING_AUTOMATIC,
    AVIF_CHROMA_DOWNSAMPLING_FASTEST,
    AVIF_CHROMA_DOWNSAMPLING_BEST_QUALITY,
    AVIF_CHROMA_DOWNSAMPLING_AVERAGE,
    AVIF_CHROMA_DOWNSAMPLING_SHARP_YUV,
} avifChromaDownsampling;

typedef enum avifChromaSamplePosition {
    AVIF_CHROMA_SAMPLE_POSITION_UNKNOWN = 0,
    AVIF_CHROMA_SAMPLE_POSITION_VERTICAL = 1,
    AVIF_CHROMA_SAMPLE_POSITION_COLOCATED = 2,
} avifChromaSamplePosition;

typedef enum avifChromaUpsampling {
    AVIF_CHROMA_UPSAMPLING_AUTOMATIC,
    AVIF_CHROMA_UPSAMPLING_FASTEST,
    AVIF_CHROMA_UPSAMPLING_BEST_QUALITY,
    AVIF_CHROMA_UPSAMPLING_NEAREST,
    AVIF_CHROMA_UPSAMPLING_BILINEAR,
} avifChromaUpsampling;

enum avifColorPrimaries
#ifdef __cplusplus
  : uint16_t
#endif // __cplusplus
 {
    AVIF_COLOR_PRIMARIES_UNKNOWN = 0,
    AVIF_COLOR_PRIMARIES_SRGB = 1,
    AVIF_COLOR_PRIMARIES_UNSPECIFIED = 2,
    AVIF_COLOR_PRIMARIES_BT470M = 4,
    AVIF_COLOR_PRIMARIES_BT470BG = 5,
    AVIF_COLOR_PRIMARIES_BT601 = 6,
    AVIF_COLOR_PRIMARIES_SMPTE240 = 7,
    AVIF_COLOR_PRIMARIES_GENERIC_FILM = 8,
    AVIF_COLOR_PRIMARIES_BT2020 = 9,
    AVIF_COLOR_PRIMARIES_XYZ = 10,
    AVIF_COLOR_PRIMARIES_SMPTE431 = 11,
    AVIF_COLOR_PRIMARIES_SMPTE432 = 12,
    AVIF_COLOR_PRIMARIES_EBU3213 = 22,
};
#ifndef __cplusplus
typedef uint16_t avifColorPrimaries;
#endif // __cplusplus

typedef enum avifRGBFormat {
    AVIF_RGB_FORMAT_RGB,
    AVIF_RGB_FORMAT_RGBA,
    AVIF_RGB_FORMAT_ARGB,
    AVIF_RGB_FORMAT_BGR,
    AVIF_RGB_FORMAT_BGRA,
    AVIF_RGB_FORMAT_ABGR,
    AVIF_RGB_FORMAT_RGB565,
} avifRGBFormat;

enum avifMatrixCoefficients
#ifdef __cplusplus
  : uint16_t
#endif // __cplusplus
 {
    AVIF_MATRIX_COEFFICIENTS_IDENTITY = 0,
    AVIF_MATRIX_COEFFICIENTS_BT709 = 1,
    AVIF_MATRIX_COEFFICIENTS_UNSPECIFIED = 2,
    AVIF_MATRIX_COEFFICIENTS_FCC = 4,
    AVIF_MATRIX_COEFFICIENTS_BT470BG = 5,
    AVIF_MATRIX_COEFFICIENTS_BT601 = 6,
    AVIF_MATRIX_COEFFICIENTS_SMPTE240 = 7,
    AVIF_MATRIX_COEFFICIENTS_YCGCO = 8,
    AVIF_MATRIX_COEFFICIENTS_BT2020_NCL = 9,
    AVIF_MATRIX_COEFFICIENTS_BT2020_CL = 10,
    AVIF_MATRIX_COEFFICIENTS_SMPTE2085 = 11,
    AVIF_MATRIX_COEFFICIENTS_CHROMA_DERIVED_NCL = 12,
    AVIF_MATRIX_COEFFICIENTS_CHROMA_DERIVED_CL = 13,
    AVIF_MATRIX_COEFFICIENTS_ICTCP = 14,
    AVIF_MATRIX_COEFFICIENTS_YCGCO_RE = 15,
    AVIF_MATRIX_COEFFICIENTS_YCGCO_RO = 16,
};
#ifndef __cplusplus
typedef uint16_t avifMatrixCoefficients;
#endif // __cplusplus

typedef enum avifProgressiveState {
    AVIF_PROGRESSIVE_STATE_UNAVAILABLE = 0,
    AVIF_PROGRESSIVE_STATE_AVAILABLE = 1,
    AVIF_PROGRESSIVE_STATE_ACTIVE = 2,
} avifProgressiveState;

typedef enum avifDecoderSource {
    AVIF_DECODER_SOURCE_AUTO = 0,
    AVIF_DECODER_SOURCE_PRIMARY_ITEM = 1,
    AVIF_DECODER_SOURCE_TRACKS = 2,
} avifDecoderSource;

enum avifTransferCharacteristics
#ifdef __cplusplus
  : uint16_t
#endif // __cplusplus
 {
    AVIF_TRANSFER_CHARACTERISTICS_UNKNOWN = 0,
    AVIF_TRANSFER_CHARACTERISTICS_BT709 = 1,
    AVIF_TRANSFER_CHARACTERISTICS_UNSPECIFIED = 2,
    AVIF_TRANSFER_CHARACTERISTICS_BT470M = 4,
    AVIF_TRANSFER_CHARACTERISTICS_BT470BG = 5,
    AVIF_TRANSFER_CHARACTERISTICS_BT601 = 6,
    AVIF_TRANSFER_CHARACTERISTICS_SMPTE240 = 7,
    AVIF_TRANSFER_CHARACTERISTICS_LINEAR = 8,
    AVIF_TRANSFER_CHARACTERISTICS_LOG100 = 9,
    AVIF_TRANSFER_CHARACTERISTICS_LOG100_SQRT10 = 10,
    AVIF_TRANSFER_CHARACTERISTICS_IEC61966 = 11,
    AVIF_TRANSFER_CHARACTERISTICS_BT1361 = 12,
    AVIF_TRANSFER_CHARACTERISTICS_SRGB = 13,
    AVIF_TRANSFER_CHARACTERISTICS_BT2020_10BIT = 14,
    AVIF_TRANSFER_CHARACTERISTICS_BT2020_12BIT = 15,
    AVIF_TRANSFER_CHARACTERISTICS_PQ = 16,
    AVIF_TRANSFER_CHARACTERISTICS_SMPTE428 = 17,
    AVIF_TRANSFER_CHARACTERISTICS_HLG = 18,
};
#ifndef __cplusplus
typedef uint16_t avifTransferCharacteristics;
#endif // __cplusplus

typedef enum avifChannelIndex {
    AVIF_CHAN_Y = 0,
    AVIF_CHAN_U = 1,
    AVIF_CHAN_V = 2,
    AVIF_CHAN_A = 3,
} avifChannelIndex;

typedef enum avifCodecChoice {
    AVIF_CODEC_CHOICE_AUTO = 0,
    AVIF_CODEC_CHOICE_AOM = 1,
    AVIF_CODEC_CHOICE_DAV1D = 2,
    AVIF_CODEC_CHOICE_LIBGAV1 = 3,
    AVIF_CODEC_CHOICE_RAV1E = 4,
    AVIF_CODEC_CHOICE_SVT = 5,
    AVIF_CODEC_CHOICE_AVM = 6,
} avifCodecChoice;

typedef enum avifCodecFlag {
    AVIF_CODEC_FLAG_CAN_DECODE = (1 << 0),
    AVIF_CODEC_FLAG_CAN_ENCODE = (1 << 1),
} avifCodecFlag;

typedef enum avifHeaderFormat {
    AVIF_HEADER_FULL,
    AVIF_HEADER_REDUCED,
} avifHeaderFormat;

typedef enum avifPixelFormat {
    AVIF_PIXEL_FORMAT_NONE,
    AVIF_PIXEL_FORMAT_YUV444,
    AVIF_PIXEL_FORMAT_YUV422,
    AVIF_PIXEL_FORMAT_YUV420,
    AVIF_PIXEL_FORMAT_YUV400,
    AVIF_PIXEL_FORMAT_COUNT,
} avifPixelFormat;

typedef enum avifPlanesFlag {
    AVIF_PLANES_YUV = (1 << 0),
    AVIF_PLANES_A = (1 << 1),
    AVIF_PLANES_ALL = 255,
} avifPlanesFlag;

typedef enum avifRange {
    AVIF_RANGE_LIMITED = 0,
    AVIF_RANGE_FULL = 1,
} avifRange;

typedef enum avifResult {
    AVIF_RESULT_OK = 0,
    AVIF_RESULT_UNKNOWN_ERROR = 1,
    AVIF_RESULT_INVALID_FTYP = 2,
    AVIF_RESULT_NO_CONTENT = 3,
    AVIF_RESULT_NO_YUV_FORMAT_SELECTED = 4,
    AVIF_RESULT_REFORMAT_FAILED = 5,
    AVIF_RESULT_UNSUPPORTED_DEPTH = 6,
    AVIF_RESULT_ENCODE_COLOR_FAILED = 7,
    AVIF_RESULT_ENCODE_ALPHA_FAILED = 8,
    AVIF_RESULT_BMFF_PARSE_FAILED = 9,
    AVIF_RESULT_MISSING_IMAGE_ITEM = 10,
    AVIF_RESULT_DECODE_COLOR_FAILED = 11,
    AVIF_RESULT_DECODE_ALPHA_FAILED = 12,
    AVIF_RESULT_COLOR_ALPHA_SIZE_MISMATCH = 13,
    AVIF_RESULT_ISPE_SIZE_MISMATCH = 14,
    AVIF_RESULT_NO_CODEC_AVAILABLE = 15,
    AVIF_RESULT_NO_IMAGES_REMAINING = 16,
    AVIF_RESULT_INVALID_EXIF_PAYLOAD = 17,
    AVIF_RESULT_INVALID_IMAGE_GRID = 18,
    AVIF_RESULT_INVALID_CODEC_SPECIFIC_OPTION = 19,
    AVIF_RESULT_TRUNCATED_DATA = 20,
    AVIF_RESULT_IO_NOT_SET = 21,
    AVIF_RESULT_IO_ERROR = 22,
    AVIF_RESULT_WAITING_ON_IO = 23,
    AVIF_RESULT_INVALID_ARGUMENT = 24,
    AVIF_RESULT_NOT_IMPLEMENTED = 25,
    AVIF_RESULT_OUT_OF_MEMORY = 26,
    AVIF_RESULT_CANNOT_CHANGE_SETTING = 27,
    AVIF_RESULT_INCOMPATIBLE_IMAGE = 28,
    AVIF_RESULT_ENCODE_GAIN_MAP_FAILED = 29,
    AVIF_RESULT_DECODE_GAIN_MAP_FAILED = 30,
    AVIF_RESULT_INVALID_TONE_MAPPED_IMAGE = 31,
} avifResult;

typedef struct Decoder Decoder;

typedef int avifBool;

typedef uint32_t avifStrictFlags;

typedef struct avifRWData {
    uint8_t *data;
    size_t size;
} avifRWData;

typedef struct ContentLightLevelInformation {
    uint16_t maxCLL;
    uint16_t maxPALL;
} ContentLightLevelInformation;

typedef struct ContentLightLevelInformation avifContentLightLevelInformationBox;

typedef uint32_t avifTransformFlags;

typedef struct PixelAspectRatio {
    uint32_t hSpacing;
    uint32_t vSpacing;
} PixelAspectRatio;

typedef struct PixelAspectRatio avifPixelAspectRatioBox;

typedef struct avifCleanApertureBox {
    uint32_t widthN;
    uint32_t widthD;
    uint32_t heightN;
    uint32_t heightD;
    uint32_t horizOffN;
    uint32_t horizOffD;
    uint32_t vertOffN;
    uint32_t vertOffD;
} avifCleanApertureBox;

typedef struct avifImageRotation {
    uint8_t angle;
} avifImageRotation;

typedef struct avifImageMirror {
    uint8_t axis;
} avifImageMirror;

typedef struct avifGainMapMetadata {
    int32_t gainMapMinN[3];
    uint32_t gainMapMinD[3];
    int32_t gainMapMaxN[3];
    uint32_t gainMapMaxD[3];
    uint32_t gainMapGammaN[3];
    uint32_t gainMapGammaD[3];
    int32_t baseOffsetN[3];
    uint32_t baseOffsetD[3];
    int32_t alternateOffsetN[3];
    uint32_t alternateOffsetD[3];
    uint32_t baseHdrHeadroomN;
    uint32_t baseHdrHeadroomD;
    uint32_t alternateHdrHeadroomN;
    uint32_t alternateHdrHeadroomD;
    avifBool backwardDirection;
    avifBool useBaseColorSpace;
} avifGainMapMetadata;

typedef struct avifGainMap {
    struct avifImage *image;
    struct avifGainMapMetadata metadata;
    struct avifRWData altICC;
    avifColorPrimaries altColorPrimaries;
    avifTransferCharacteristics altTransferCharacteristics;
    avifMatrixCoefficients altMatrixCoefficients;
    enum avifRange altYUVRange;
    uint32_t altDepth;
    uint32_t altPlaneCount;
    avifContentLightLevelInformationBox altCLLI;
} avifGainMap;

typedef struct avifImage {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    enum avifPixelFormat yuvFormat;
    enum avifRange yuvRange;
    enum avifChromaSamplePosition yuvChromaSamplePosition;
    uint8_t *yuvPlanes[CRABBY_AVIF_PLANE_COUNT_YUV];
    uint32_t yuvRowBytes[CRABBY_AVIF_PLANE_COUNT_YUV];
    avifBool imageOwnsYUVPlanes;
    uint8_t *alphaPlane;
    uint32_t alphaRowBytes;
    avifBool imageOwnsAlphaPlane;
    avifBool alphaPremultiplied;
    struct avifRWData icc;
    avifColorPrimaries colorPrimaries;
    avifTransferCharacteristics transferCharacteristics;
    avifMatrixCoefficients matrixCoefficients;
    avifContentLightLevelInformationBox clli;
    avifTransformFlags transformFlags;
    avifPixelAspectRatioBox pasp;
    struct avifCleanApertureBox clap;
    struct avifImageRotation irot;
    struct avifImageMirror imir;
    struct avifRWData exif;
    struct avifRWData xmp;
    struct avifGainMap *gainMap;
} avifImage;

typedef struct avifImageTiming {
    uint64_t timescale;
    double pts;
    uint64_t ptsInTimescales;
    double duration;
    uint64_t durationInTimescales;
} avifImageTiming;

typedef struct avifIOStats {
    size_t colorOBUSize;
    size_t alphaOBUSize;
} avifIOStats;

typedef struct avifDiagnostics {
    char error[CRABBY_AVIF_DIAGNOSTICS_ERROR_BUFFER_SIZE];
} avifDiagnostics;

typedef struct avifDecoderData {

} avifDecoderData;

typedef struct avifDecoder {
    enum avifCodecChoice codecChoice;
    int32_t maxThreads;
    enum avifDecoderSource requestedSource;
    avifBool allowProgressive;
    avifBool allowIncremental;
    avifBool ignoreExif;
    avifBool ignoreXMP;
    uint32_t imageSizeLimit;
    uint32_t imageDimensionLimit;
    uint32_t imageCountLimit;
    avifStrictFlags strictFlags;
    struct avifImage *image;
    int32_t imageIndex;
    int32_t imageCount;
    enum avifProgressiveState progressiveState;
    struct avifImageTiming imageTiming;
    uint64_t timescale;
    double duration;
    uint64_t durationInTimescales;
    int32_t repetitionCount;
    avifBool alphaPresent;
    struct avifIOStats ioStats;
    struct avifDiagnostics diag;
    struct avifDecoderData *data;
    avifBool gainMapPresent;
    avifBool enableDecodingGainMap;
    avifBool enableParsingGainMapMetadata;
    avifBool imageSequenceTrackPresent;
    struct Decoder *rust_decoder;
    struct avifImage image_object;
    struct avifGainMap gainmap_object;
    struct avifImage gainmap_image_object;
} avifDecoder;

typedef void (*avifIODestroyFunc)(struct avifIO *io);

typedef struct avifROData {
    const uint8_t *data;
    size_t size;
} avifROData;

typedef enum avifResult (*avifIOReadFunc)(struct avifIO *io,
                                          uint32_t readFlags,
                                          uint64_t offset,
                                          size_t size,
                                          struct avifROData *out);

typedef enum avifResult (*avifIOWriteFunc)(struct avifIO *io,
                                           uint32_t writeFlags,
                                           uint64_t offset,
                                           const uint8_t *data,
                                           size_t size);

typedef struct avifIO {
    avifIODestroyFunc destroy;
    avifIOReadFunc read;
    avifIOWriteFunc write;
    uint64_t sizeHint;
    avifBool persistent;
    void *data;
} avifIO;

typedef struct Extent {
    uint64_t offset;
    size_t size;
} Extent;

typedef struct Extent avifExtent;

typedef uint32_t avifPlanesFlags;

typedef struct CropRect {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
} CropRect;

typedef struct CropRect avifCropRect;

typedef struct avifRGBImage {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    enum avifRGBFormat format;
    enum avifChromaUpsampling chromaUpsampling;
    enum avifChromaDownsampling chromaDownsampling;
    bool ignoreAlpha;
    bool alphaPremultiplied;
    bool isFloat;
    int32_t maxThreads;
    uint8_t *pixels;
    uint32_t rowBytes;
} avifRGBImage;

typedef struct avifPixelFormatInfo {
    avifBool monochrome;
    int chromaShiftX;
    int chromaShiftY;
} avifPixelFormatInfo;

typedef uint32_t avifCodecFlags;











#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct avifDecoder *crabby_avifDecoderCreate(void);

void crabby_avifDecoderSetIO(struct avifDecoder *decoder, struct avifIO *io);

enum avifResult crabby_avifDecoderSetIOFile(struct avifDecoder *decoder, const char *filename);

enum avifResult crabby_avifDecoderSetIOMemory(struct avifDecoder *decoder,
                                              const uint8_t *data,
                                              size_t size);

enum avifResult crabby_avifDecoderSetSource(struct avifDecoder *decoder,
                                            enum avifDecoderSource source);

enum avifResult crabby_avifDecoderParse(struct avifDecoder *decoder);

enum avifResult crabby_avifDecoderNextImage(struct avifDecoder *decoder);

enum avifResult crabby_avifDecoderNthImage(struct avifDecoder *decoder, uint32_t frameIndex);

enum avifResult crabby_avifDecoderNthImageTiming(const struct avifDecoder *decoder,
                                                 uint32_t frameIndex,
                                                 struct avifImageTiming *outTiming);

void crabby_avifDecoderDestroy(struct avifDecoder *decoder);

enum avifResult crabby_avifDecoderRead(struct avifDecoder *decoder, struct avifImage *image);

enum avifResult crabby_avifDecoderReadMemory(struct avifDecoder *decoder,
                                             struct avifImage *image,
                                             const uint8_t *data,
                                             size_t size);

enum avifResult crabby_avifDecoderReadFile(struct avifDecoder *decoder,
                                           struct avifImage *image,
                                           const char *filename);

avifBool crabby_avifDecoderIsKeyframe(const struct avifDecoder *decoder, uint32_t frameIndex);

uint32_t crabby_avifDecoderNearestKeyframe(const struct avifDecoder *decoder, uint32_t frameIndex);

uint32_t crabby_avifDecoderDecodedRowCount(const struct avifDecoder *decoder);

enum avifResult crabby_avifDecoderNthImageMaxExtent(const struct avifDecoder *decoder,
                                                    uint32_t frameIndex,
                                                    avifExtent *outExtent);

avifBool crabby_avifPeekCompatibleFileType(const struct avifROData *input);

struct avifImage *crabby_avifImageCreateEmpty(void);

struct avifImage *crabby_avifImageCreate(uint32_t width,
                                         uint32_t height,
                                         uint32_t depth,
                                         enum avifPixelFormat yuvFormat);

enum avifResult crabby_avifImageAllocatePlanes(struct avifImage *image, avifPlanesFlags planes);

void crabby_avifImageFreePlanes(struct avifImage *image, avifPlanesFlags planes);

void crabby_avifImageDestroy(struct avifImage *image);

avifBool crabby_avifImageUsesU16(const struct avifImage *image);

avifBool crabby_avifImageIsOpaque(const struct avifImage *image);

uint8_t *crabby_avifImagePlane(const struct avifImage *image, int channel);

uint32_t crabby_avifImagePlaneRowBytes(const struct avifImage *image, int channel);

uint32_t crabby_avifImagePlaneWidth(const struct avifImage *image, int channel);

uint32_t crabby_avifImagePlaneHeight(const struct avifImage *image, int channel);

enum avifResult crabby_avifImageSetViewRect(struct avifImage *dstImage,
                                            const struct avifImage *srcImage,
                                            const avifCropRect *rect);

enum avifResult crabby_avifRWDataRealloc(struct avifRWData *raw, size_t newSize);

enum avifResult crabby_avifRWDataSet(struct avifRWData *raw, const uint8_t *data, size_t size);

void crabby_avifRWDataFree(struct avifRWData *raw);

void cioDestroy(struct avifIO *_io);

enum avifResult cioRead(struct avifIO *io,
                        uint32_t _readFlags,
                        uint64_t offset,
                        size_t size,
                        struct avifROData *out);

enum avifResult cioWrite(struct avifIO *_io,
                         uint32_t _writeFlags,
                         uint64_t _offset,
                         const uint8_t *_data,
                         size_t _size);

struct avifIO *crabby_avifIOCreateMemoryReader(const uint8_t *data, size_t size);

struct avifIO *crabby_avifIOCreateFileReader(const char *filename);

void crabby_avifIODestroy(struct avifIO *io);

void crabby_avifRGBImageSetDefaults(struct avifRGBImage *rgb, const struct avifImage *image);

enum avifResult crabby_avifImageYUVToRGB(const struct avifImage *image, struct avifRGBImage *rgb);

const char *crabby_avifResultToString(enum avifResult _res);

avifBool crabby_avifCropRectConvertCleanApertureBox(avifCropRect *cropRect,
                                                    const struct avifCleanApertureBox *clap,
                                                    uint32_t imageW,
                                                    uint32_t imageH,
                                                    enum avifPixelFormat yuvFormat,
                                                    struct avifDiagnostics *_diag);

void crabby_avifGetPixelFormatInfo(enum avifPixelFormat format, struct avifPixelFormatInfo *info);

void crabby_avifDiagnosticsClearError(struct avifDiagnostics *diag);

void *crabby_avifAlloc(size_t size);

void crabby_avifFree(void *p);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif /* AVIF_H */

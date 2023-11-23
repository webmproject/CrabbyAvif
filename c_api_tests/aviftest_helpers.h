#include <memory>

#include "avif/avif.h"

namespace avif {

// Struct to call the destroy functions in a unique_ptr.
struct UniquePtrDeleter
{
    //void operator()(avifEncoder * encoder) const { avifEncoderDestroy(encoder); }
    void operator()(avifDecoder * decoder) const { avifDecoderDestroy(decoder); }
    //void operator()(avifImage * image) const { avifImageDestroy(image); }
};

// Use these unique_ptr to ensure the structs are automatically destroyed.
//using EncoderPtr = std::unique_ptr<avifEncoder, UniquePtrDeleter>;
using DecoderPtr = std::unique_ptr<avifDecoder, UniquePtrDeleter>;
//using ImagePtr = std::unique_ptr<avifImage, UniquePtrDeleter>;

}

namespace testutil {
    bool Av1DecoderAvailable() { return true; }
}
//
// Created by fotoable on 2017/8/7.
//

#include "interp.h"
#include <assert.h>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON
namespace ncnn {
    DEFINE_LAYER_CREATOR(Interp);

    Interp::Interp() {
        one_blob_only = true;
    }

    Interp::~Interp() {};

#if NCNN_STDIO
#if NCNN_STRING

    int Interp::load_param(FILE *paramfp) {
        int nscan = fscanf(paramfp, "%d %f %f %d %d", &type, &height_scale_, &width_scale_,&output_width_,&output_height_);
        if (nscan != 3) {
            fprintf(stderr, "Interp load_param failed %d\n", nscan);
            return -1;
        }
        return 0;
    }

#endif

    int Interp::load_param_bin(FILE *paramfp) {
        fread(&resize_type_, sizeof(int), 1, paramfp);
        fread(&height_scale_, sizeof(float), 1, paramfp);
        fread(&width_scale_, sizeof(float), 1, paramfp);
        fread(&output_height_, sizeof(int), 1, paramfp);
        fread(&output_width_, sizeof(int), 1, paramfp);
        return 0;
    }

#endif

    int Interp::load_param(const unsigned char *&mem) {
        resize_type_ = *(int *) (mem);
        mem += 4;
        height_scale_ = *(float *) (mem);
        mem += 4;
        width_scale_ = *(float *) (mem);
        mem += 4;
        output_height_ = *(float *) (mem);
        mem += 4;
        output_width_ = *(float *) (mem);
        mem += 4;
        return 0;
    }

    void resizeNearest2x(
            int N,
            int C,
            int H,
            int W,
            const float *input,
            float *output) {
        const int outputH = H * 2;
        const int outputW = W * 2;
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int y = 0; y < outputH; ++y) {
                    const int y_in = y / 2;
#ifdef __ARM_NEON
                    int vecW = (W / 4) * 4; // round down
        int x = 0;
        for (; x < vecW; x += 4) {
          // load 0 1 2 3
          float32x4_t v = vld1q_f32(input + y_in * W + x);
          const int oidx = outputW * y + x * 2;
          float32x4x2_t v2 = {{v, v}};
          // store 00 11 22 33
          vst2q_f32(output + oidx + 0, v2);
        }

        // handle remainder
        for (; x < W; ++x) {
          const float v = input[y_in * W + x];
          const int oidx = outputW * y + x * 2;
          output[oidx + 0] = v;
          output[oidx + 1] = v;
        }
#else
                    for (int x = 0; x < W; ++x) {
                        const float v = input[y_in * W + x];
                        const int oidx = outputW * y + x * 2;
                        output[oidx + 0] = v;
                        output[oidx + 1] = v;
                    }
#endif
                }
                input += H * W;
                output += outputH * outputW;
            }
        }
    }

    void bilinear_cpu_interp2(const int channels,
                           const float *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
                           float *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
        assert(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
        assert(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2);
        // special case: just copy
        if (height1 == height2 && width1 == width2) {
            for (int h2 = 0; h2 < height2; ++h2) {
                const int h1 = h2;
                for (int w2 = 0; w2 < width2; ++w2) {
                    const int w1 = w2;
                    const float* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
                    float* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
                    for (int c = 0; c < channels; ++c) {
                        pos2[0] = pos1[0];
                        pos1 += Width1 * Height1;
                        pos2 += Width2 * Height2;
                    }
                }
            }
            return;
        }
        const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
        const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
        for (int h2 = 0; h2 < height2; ++h2) {
            const float h1r = rheight * h2;
            const int h1 = h1r;
            const int h1p = (h1 < height1 - 1) ? 1 : 0;
            const float h1lambda = h1r - h1;
            const float h0lambda = float(1.) - h1lambda;
            for (int w2 = 0; w2 < width2; ++w2) {
                const float w1r = rwidth * w2;
                const int w1 = w1r;
                const int w1p = (w1 < width1 - 1) ? 1 : 0;
                const float w1lambda = w1r - w1;
                const float w0lambda = float(1.) - w1lambda;
                const float* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
                float* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
                for (int c = 0; c < channels; ++c) {
                    pos2[0] =
                            h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[w1p]) +
                            h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
                    pos1 += Width1 * Height1;
                    pos2 += Width2 * Height2;
                }
            }
        }
    }
    int Interp::forward(const Mat &bottom_blobs, Mat &top_blobs) const {
        auto h = bottom_blobs.h;
        auto w = bottom_blobs.w;
        auto c = bottom_blobs.c;
        auto output_height = output_height_;
        auto output_width = output_width_;
        if (output_width_ == 0 || output_width_ == 0) {
            output_height = h * height_scale_;
            output_width = w * width_scale_;
        }
        top_blobs.reshape(output_width, output_height, c);
        if (resize_type_ == 1)//nearneast
        {
            if ((width_scale_ == 2.0 && height_scale_ == 2.0) || (output_height / h == 2 && output_width / w == 2)) {
                resizeNearest2x(1, c, h, w, bottom_blobs.data, top_blobs.data);
                return 0;
            }
            auto bottom_ptr = bottom_blobs.data;
            for (int n = 0; n < 1; ++n) {
                for (int c = 0; c < c; ++c) {
                    for (int y = 0; y < output_height; ++y) {
                        const int in_y = std::min((int) (y / height_scale_), (h - 1));
                        for (int x = 0; x < output_width; ++x) {
                            const int in_x = std::min((int) (x / width_scale_), (w - 1));
                            top_blobs.data[output_width * y + x] = bottom_ptr[in_y * w + in_x];
                        }
                    }
                    bottom_ptr += h * w;
                    top_blobs.data += output_width * output_height;
                }
            }
            return 0;
        }
        else if (resize_type_ == 2)// bilinear
        {
            bilinear_cpu_interp2(c,bottom_blobs.data,0,0,h,w,h,w,top_blobs.data,0,0,output_height,output_width,output_height,output_width);
            return 0;
        }
        else{
            fprintf(stderr, "unsupported resize type\n");
            return -1;
        }
    }
}

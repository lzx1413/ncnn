// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "concat.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Concat)

Concat::Concat()
{
}
#if NCNN_STDIO
#if NCNN_STRING

    int Concat::load_param(FILE *paramfp) {
        int nscan = fscanf(paramfp, " %d", &axis);
        if (nscan != 1) {
            fprintf(stderr, "Concat load_param failed %d\n", nscan);
            return -1;
        }
        return 0;
    }

#endif

    int Concat::load_param_bin(FILE *paramfp) {
        fread(&axis, sizeof(int), 1, paramfp);
        return 0;
    }

#endif

    int Concat::load_param(const unsigned char *&mem) {
        axis = *(int *) (mem);
        mem += 4;
        return 0;
    }

int Concat::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    int dims = bottom_blobs[0].dims;
    if (dims == 1) {
        // concat vector
        // total length
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            const Mat &bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat &top_blob = top_blobs[0];
        top_blob.create(top_w);
        if (top_blob.empty())
            return -100;

        float *outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            const Mat &bottom_blob = bottom_blobs[b];

            int w = bottom_blob.w;

            const float *ptr = bottom_blob;
            memcpy(outptr,ptr,w* sizeof(float));
            outptr += w;
        }
        return 0;
    }

    int h = bottom_blobs[0].h;
    int c = bottom_blobs[0].c;
    int w = bottom_blobs[0].w;
    if(axis == 1) {
        // total channels
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            const Mat &bottom_blob = bottom_blobs[b];
            top_channels += bottom_blob.c;
        }

        Mat &top_blob = top_blobs[0];
        top_blob.create(w, h, top_channels);
        if (top_blob.empty())
            return -100;

        int q = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            const Mat &bottom_blob = bottom_blobs[b];

            int channels = bottom_blob.c;
            int size = bottom_blob.cstep * channels;

            const float *ptr = bottom_blob;
            float *outptr = top_blob.channel(q);
            memcpy(outptr,ptr,size* sizeof(float));
            q += channels;
        }
    }
    else if (axis == 2)
    {
        //total heights
        int top_height = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            const Mat &bottom_blob = bottom_blobs[b];
            top_height += bottom_blob.h;
        }
        Mat &top_blob = top_blobs[0];
        top_blob.create(w, top_height, c);
        if (top_blob.empty())
            return -100;
#pragma omp parallel for
        for (int q=0; q<c; q++)
        {
            float* outptr = top_blob.channel(q);
            for (size_t b = 0;b<bottom_blobs.size();b++)
            {
                const Mat & bottom_blob = bottom_blobs.at(b);
                const float* ptr = bottom_blob.channel(q);
                memcpy(outptr,ptr,bottom_blob.cstep* sizeof(float));
                outptr += bottom_blob.cstep;

            }
        }
    }
    else if(axis == 3) {
        // total widths
        int top_width = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            const Mat &bottom_blob = bottom_blobs[b];
            top_width += bottom_blob.w;
        }

        Mat &top_blob = top_blobs[0];
        top_blob.create(top_width, h, c);

        if (top_blob.empty())
            return -100;

        int cur_w = 0;

        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            const Mat &bottom_blob = bottom_blobs[b];
            const float *ptr = bottom_blob;
            int w = bottom_blob.w;
            float *outptr = top_blob;
            for (int ic = 0; ic < c; ic++)
                for (int ih = 0; ih < h; ih++)
                    for (int iw = 0; iw < w; iw++) {
                        outptr[cur_w + iw + ih * top_width + ic * top_width * h] = ptr[iw + ih * w + ic * h * w];
                    }
            cur_w += w;
        }
    }
    return 0;
}

} // namespace ncnn

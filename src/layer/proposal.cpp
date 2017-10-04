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

#include "proposal.h"
#include "bbox_util.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(Proposal)


Proposal::Proposal()
{
}

#if NCNN_STDIO
#if NCNN_STRING
int Proposal::load_param(FILE* paramfp)
{
//     float ratio;
//     float scale;
    int nscan = fscanf(paramfp, "%d %d %d %d %f %d",
                       &feat_stride, &base_size, &pre_nms_topN, &after_nms_topN,
                       &nms_thresh, &min_size);
    if (nscan != 6)
    {
        fprintf(stderr, "Proposal load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int Proposal::load_param_bin(FILE* paramfp)
{
    fread(&feat_stride, sizeof(int), 1, paramfp);

    fread(&base_size, sizeof(int), 1, paramfp);

//     float ratio;
//     float scale;

    fread(&pre_nms_topN, sizeof(int), 1, paramfp);

    fread(&after_nms_topN, sizeof(int), 1, paramfp);

    fread(&nms_thresh, sizeof(float), 1, paramfp);

    fread(&min_size, sizeof(int), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int Proposal::load_param(const unsigned char*& mem)
{
    feat_stride = *(int*)(mem);
    mem += 4;

    base_size = *(int*)(mem);
    mem += 4;

//     float ratio;
//     float scale;

    pre_nms_topN = *(int*)(mem);
    mem += 4;

    after_nms_topN = *(int*)(mem);
    mem += 4;

    nms_thresh = *(float*)(mem);
    mem += 4;

    min_size = *(int*)(mem);
    mem += 4;

    return 0;
}

int Proposal::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    const Mat& score_blob = bottom_blobs[0];
    const Mat& bbox_blob = bottom_blobs[1];
    const Mat& im_info_blob = bottom_blobs[2];

    int w = score_blob.w;
    int h = score_blob.h;

    // for each (H, W) location i
    // generate A anchor boxes centered on cell i
    // apply predicted bbox deltas at cell i to each of the A anchors
    Rect base_anchor(0, 0, base_size - 1, base_size - 1);

    // generate all ratio anchors
    float ratios[3] = { 0.5, 1, 2 };
    Rect ratio_anchors[3];
    {
        int size = base_anchor.area();
        float cx = base_anchor.x + 0.5f * base_anchor.width;
        float cy = base_anchor.y + 0.5f * base_anchor.height;
        for (int i=0; i<3; i++)
        {
            float aw = (int)(sqrt(size / ratios[i]) + 0.5f);
            float ah = (int)(aw * ratios[i] + 0.5f);
            float ax = cx - 0.5f * (aw - 1);
            float ay = cy - 0.5f * (ah - 1);
            ratio_anchors[i] = Rect(ax, ay, aw, ah);
        }
    }

    // generate all scale anchors
    float scales[3] = { 8, 16, 32 };
    Rect anchors[3*3];
    {
        for (int i=0; i<3; i++)
        {
            const Rect& ra = ratio_anchors[i];
            float cx = ra.x + 0.5f * ra.width;
            float cy = ra.y + 0.5f * ra.height;
            for (int j=0; j<3; j++)
            {
                float aw = ra.width * scales[j];
                float ah = ra.height * scales[j];
                float ax = cx - 0.5f * (aw - 1);
                float ay = cy - 0.5f * (ah - 1);
                anchors[i*3+j] = Rect(ax, ay, aw, ah);
            }
        }
    }

    // generate proposals from bbox deltas and shifted anchors
    // clip predicted boxes to image
    std::vector<Rect > proposals;
    int num_anchors = 3*3;
    proposals.resize(num_anchors * h * w);

    float im_w = ((const float*)im_info_blob.data)[1];
    float im_h = ((const float*)im_info_blob.data)[0];

    #pragma omp parallel for
    for (int k = 0; k < num_anchors; k++)
    {
        const float* bbox_xptr = (const float*)(bbox_blob.data + bbox_blob.cstep * (k * 4 + 0));
        const float* bbox_yptr = (const float*)(bbox_blob.data + bbox_blob.cstep * (k * 4 + 1));
        const float* bbox_wptr = (const float*)(bbox_blob.data + bbox_blob.cstep * (k * 4 + 2));
        const float* bbox_hptr = (const float*)(bbox_blob.data + bbox_blob.cstep * (k * 4 + 3));

        // shifted anchor
        Rect sa = anchors[k];
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                // apply bbox deltas
                float dx = bbox_xptr[j];
                float dy = bbox_yptr[j];
                float dw = bbox_wptr[j];
                float dh = bbox_hptr[j];

                float cx = sa.x + 0.5f * sa.width;
                float cy = sa.y + 0.5f * sa.height;

                cx += sa.width * dx;
                cy += sa.height * dy;
                float aw = sa.width * exp(dw);
                float ah = sa.height * exp(dh);
                float ax = cx - 0.5f * aw;
                float ay = cy - 0.5f * ah;

                // clip box
                ax = std::max(std::min(ax, im_w - 1), 0.f);
                ay = std::max(std::min(ay, im_h - 1), 0.f);
                aw = std::max(std::min(aw, im_w - ax), 0.f);
                ah = std::max(std::min(ah, im_h - ay), 0.f);

                proposals[k * h * w + i * w + j] = Rect(ax, ay, aw, ah);

                sa.x += feat_stride;
            }

            bbox_xptr += w;
            bbox_yptr += w;
            bbox_wptr += w;
            bbox_hptr += w;

            sa.x = anchors[k].x;
            sa.y += feat_stride;
        }
    }

    // remove predicted boxes with either height or width < threshold
    // NOTE convert min_size to input image scale stored in im_info[2]
    std::vector<ProposalBox> proposal_boxes;

    float im_scale = ((const float*)im_info_blob.data)[2];
    float min_boxsize = min_size * im_scale;

    const float* scoreptr = (const float*)(score_blob.data);
    for (size_t i=0; i<proposals.size(); i++)
    {
        const Rect& p = proposals[i];
        if (p.width >= min_boxsize && p.height >= min_boxsize)
        {
            ProposalBox pb;
            pb.box = p;
            pb.score = scoreptr[i];
            proposal_boxes.push_back(pb);
        }
    }
    proposals.clear();

    // sort all (proposal, score) pairs by score from highest to lowest
    std::sort(proposal_boxes.begin(), proposal_boxes.end());

    // take top pre_nms_topN
    if (pre_nms_topN > 0 && pre_nms_topN < (int)proposal_boxes.size())
        proposal_boxes.resize(pre_nms_topN);

    // apply nms with nms_thresh
    std::vector<int> picked = nms(proposal_boxes, nms_thresh);

    // take after_nms_topN
    int picked_count = std::min((int)picked.size(), after_nms_topN);

    // return the top proposals (-> RoIs top)
    Mat& roi_blob = top_blobs[0];
    roi_blob.create(4, picked_count, 1);
    if (roi_blob.empty())
        return -100;

    float* outptr = roi_blob;
    for (int i=0; i<picked_count; i++)
    {
        outptr[0] = proposal_boxes[ picked[i] ].box.x;
        outptr[1] = proposal_boxes[ picked[i] ].box.y;
        outptr[2] = proposal_boxes[ picked[i] ].box.width;
        outptr[3] = proposal_boxes[ picked[i] ].box.height;

        outptr += 4;
    }

    if (top_blobs.size() > 1)
    {
        Mat& roi_score_blob = top_blobs[1];
        roi_score_blob.create(picked_count, 1, 1);
        if (roi_score_blob.empty())
            return -100;

        float* outptr = roi_score_blob;
        for (int i=0; i<picked_count; i++)
        {
            outptr[i] = proposal_boxes[ picked[i] ].score;
        }
    }

    return 0;
}

} // namespace ncnn

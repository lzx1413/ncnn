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

#include "detectionoutput.h"
#include <math.h>
#include <algorithm>
#include <vector>
#include <map>
#include "utils/bbox_util.h"

using namespace std;

#define CHECK_GT(x,y) (x)>(y)?true:false
#define CHECK_EQ(x,y) (x)==(y)?true:false
#define CHECK_LT(x,y) (x)<(y)?true:false

namespace ncnn {

    DEFINE_LAYER_CREATOR(DetectionOutput)

    DetectionOutput::DetectionOutput()
    {
        one_blob_only = false;
        support_inplace = false;
    }

#if NCNN_STDIO
#if NCNN_STRING
    int DetectionOutput::load_param(FILE* paramfp)
    {
        int nscan = fscanf(paramfp, "%d %f %d %d %f",
                           &num_classes, &nms_threshold, &nms_top_k, &keep_top_k, &confidence_threshold);
        if (nscan != 5)
        {
            fprintf(stderr, "DetectionOutput load_param failed %d\n", nscan);
            return -1;
        }
        return 0;
    }
#endif // NCNN_STRING
    int DetectionOutput::load_param_bin(FILE* paramfp)
    {
        fread(&num_classes, sizeof(int), 1, paramfp);
        fread(&nms_threshold, sizeof(float), 1, paramfp);
        fread(&nms_top_k, sizeof(int), 1, paramfp);
        fread(&keep_top_k, sizeof(int), 1, paramfp);
        fread(&confidence_threshold, sizeof(float), 1, paramfp);

        return 0;
    }
#endif // NCNN_STDIO

    int DetectionOutput::load_param(const unsigned char*& mem)
    {
        num_classes = *(int*)(mem);
        mem += 4;

        nms_threshold = *(float*)(mem);
        mem += 4;

        nms_top_k = *(int*)(mem);
        mem += 4;

        keep_top_k = *(int*)(mem);
        mem += 4;

        confidence_threshold = *(float*)(mem);
        mem += 4;

        return 0;
    }


    int DetectionOutput::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
    {
        const Dtype* loc_data = bottom_blobs[0];
        const Dtype* conf_data = bottom_blobs[1];
        const Dtype* prior_data = bottom_blobs[2];
        const int num = 1; // only one image

        int num_priors_ = bottom_blobs[0].w/4;
        bool share_location_ = true;
        int num_classes_ = num_classes;
        int num_loc_classes_ = share_location_ ? 1 : num_classes_;
        int background_label_id_ = 0;
        CodeType code_type_ = PriorBoxParameter_CodeType_CENTER_SIZE;
        float confidence_threshold_ = confidence_threshold;
        float nms_threshold_ = nms_threshold;
        int top_k_ = nms_top_k;
        int keep_top_k_ = keep_top_k;
        bool variance_encoded_in_target_ = false;
        int eta_ = 1;

        // Retrieve all location predictions.
        vector<LabelBBox> all_loc_preds;
        GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
                          share_location_, &all_loc_preds);

        // Retrieve all confidences.
        vector<map<int, vector<float> > > all_conf_scores;
        GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
                            &all_conf_scores);

        // Retrieve all prior bboxes. It is same within a batch since we assume all
        // images in a batch are of same dimension.
        vector<NormalizedBBox> prior_bboxes;
        vector<vector<float> > prior_variances;
        GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

        // Decode all loc predictions to bboxes.
        vector<LabelBBox> all_decode_bboxes;
        const bool clip_bbox = false;
        DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
                        share_location_, num_loc_classes_, background_label_id_,
                        code_type_, variance_encoded_in_target_, clip_bbox,
                        &all_decode_bboxes);

        int num_kept = 0;
        vector<map<int, vector<int> > > all_indices;
        for (int i = 0; i < num; ++i) {
            const LabelBBox& decode_bboxes = all_decode_bboxes[i];
            const map<int, vector<float> >& conf_scores = all_conf_scores[i];
            map<int, vector<int> > indices;
            int num_det = 0;
            for (int c = 0; c < num_classes_; ++c) {
                if (c == background_label_id_) {
                    // Ignore background class.
                    continue;
                }
                if (conf_scores.find(c) == conf_scores.end()) {
                    // Something bad happened if there are no predictions for current label.
                    fprintf(stderr, "Could not find confidence predictions for label %d\n",c);
                }
                const vector<float>& scores = conf_scores.find(c)->second;
                int label = share_location_ ? -1 : c;
                if (decode_bboxes.find(label) == decode_bboxes.end()) {
                    // Something bad happened if there are no predictions for current label.
                    fprintf(stderr,  "Could not find location predictions for label %d\n",label);
                    continue;
                }
                const vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
                ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_, eta_,
                             top_k_, &(indices[c]));
                num_det += indices[c].size();
            }
            if (keep_top_k_ > -1 && num_det > keep_top_k_) {
                vector<pair<float, pair<int, int> > > score_index_pairs;
                for (map<int, vector<int> >::iterator it = indices.begin();
                     it != indices.end(); ++it) {
                    int label = it->first;
                    const vector<int>& label_indices = it->second;
                    if (conf_scores.find(label) == conf_scores.end()) {
                        // Something bad happened for current label.
                        fprintf(stderr, "Could not find location predictions for %d\n",label);
                        continue;
                    }
                    const vector<float>& scores = conf_scores.find(label)->second;
                    for (int j = 0; j < label_indices.size(); ++j) {
                        int idx = label_indices[j];
                        CHECK_LT(idx, scores.size());
                        score_index_pairs.push_back(std::make_pair(
                                scores[idx], std::make_pair(label, idx)));
                    }
                }
                // Keep top k results per image.
                std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                          SortScorePairDescend<std::pair<int, int> >);
                score_index_pairs.resize(keep_top_k_);
                // Store the new indices.
                map<int, vector<int> > new_indices;
                for (int j = 0; j < score_index_pairs.size(); ++j) {
                    int label = score_index_pairs[j].second.first;
                    int idx = score_index_pairs[j].second.second;
                    new_indices[label].push_back(idx);
                }
                all_indices.push_back(new_indices);
                num_kept += keep_top_k_;
            } else {
                all_indices.push_back(indices);
                num_kept += num_det;
            }
        }

        vector<int> top_shape(2, 1);
        top_shape.push_back(num_kept);
        top_shape.push_back(7);

        //printf("%d %d %d %d %d\n", num_kept, top_shape[0], top_shape[1], top_shape[2], top_shape[3]);
        Dtype* top_data;
        if (num_kept == 0) {
            fprintf(stderr, "Couldn't find any detections\n");
            top_shape[2] = num;
            Mat& top_blob = top_blobs[0];
            top_blob.create(top_shape[1], top_shape[2], top_shape[3]);
            top_data = top_blob;
            // top[0]->Reshape(top_shape);
            // top_data = top[0]->mutable_cpu_data();
            // caffe_set<Dtype>(top[0]->count(), -1, top_data);
            // Generate fake results per image.
            for (int i = 0; i < num; ++i) {
                top_data[0] = i;
                top_data += 7;
            }
            //printf("top_data: %d %d %d", top_blob.w, top_blob.h, top_blob.c);
        } else {
            // top[0]->Reshape(top_shape);
            // top_data = top[0]->mutable_cpu_data();
            Mat& top_blob = top_blobs[0];
            top_blob.create(top_shape[3], top_shape[2]);
            top_data = top_blob;
            //printf("top_data: %d %d %d", top_blob.w, top_blob.h, top_blob.c);
        }

        int count = 0;
        int i = 0;
        for (int i = 0; i < num; ++i) {
            const map<int, vector<float> >& conf_scores = all_conf_scores[i];
            const LabelBBox& decode_bboxes = all_decode_bboxes[i];
            for (map<int, vector<int> >::iterator it = all_indices[i].begin();
                 it != all_indices[i].end(); ++it) {
                int label = it->first;
                if (conf_scores.find(label) == conf_scores.end()) {
                    // Something bad happened if there are no predictions for current label.
                    fprintf(stderr, "Could not find confidence predictions for %d\n");
                    continue;
                }
                const vector<float>& scores = conf_scores.find(label)->second;
                int loc_label = share_location_ ? -1 : label;
                if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
                    // Something bad happened if there are no predictions for current label.
                    fprintf(stderr, "Could not find location predictions for %d\n", loc_label);
                    continue;
                }
                const vector<NormalizedBBox>& bboxes =
                        decode_bboxes.find(loc_label)->second;
                vector<int>& indices = it->second;
                //printf("%d\n", indices.size());
                for (int j = 0; j < indices.size(); ++j) {
                    int idx = indices[j];
                    top_data[count * 7] = i;
                    top_data[count * 7 + 1] = label;
                    top_data[count * 7 + 2] = scores[idx];
                    const NormalizedBBox& bbox = bboxes[idx];
                    top_data[count * 7 + 3] = bbox.xmin();
                    top_data[count * 7 + 4] = bbox.ymin();
                    top_data[count * 7 + 5] = bbox.xmax();
                    top_data[count * 7 + 6] = bbox.ymax();
                    ++count;
                    //printf("%d %d %f %f %f %f %f \n",i, label, scores[idx], bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax());
                }
            }
        }
        //printf("\n");
        return 0;
    }

} // namespace ncnn
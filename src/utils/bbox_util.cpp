//
// Created by lzx1413 on 2017/9/27.
//

#include "bbox_util.h"
#include <math.h>
#include <algorithm>
#include <vector>
#include <map>
using namespace std;
#define CHECK_GT(x,y) (x)>(y)?true:false
#define CHECK_EQ(x,y) (x)==(y)?true:false
#define CHECK_LT(x,y) (x)<(y)?true:false
std::vector<int> nms(const std::vector<ProposalBox>& boxes, float nms_thresh)
{
    // NOTE boxes is already sorted
    int size = boxes.size();

    std::vector<float> areas;
    areas.resize(size);
    for (int i=0; i<size; i++)
    {
        areas[i] = boxes[i].area();
    }

    std::vector<int> suppressed;
    suppressed.resize(size, 0);

    std::vector<int> picked;

    for (int i=0; i<size; i++)
    {
        if (suppressed[i] == 1)
            continue;

        picked.push_back(i);

        for (int j=i+1; j<size; j++)
        {
            if (suppressed[j] == 1)
                continue;

            float intersize = boxes[i].inter_area(boxes[j]);
            float ov = intersize / (areas[i] + areas[j] - intersize);
            if (ov > nms_thresh)
            {
                suppressed[j] = 1;
            }
        }
    }

    return picked;
}
typedef float Dtype;
typedef map<int, std::vector<NormalizedBBox> > LabelBBox;

void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox) {
    if (bbox2.xmin() > bbox1.xmax() || bbox2.xmax() < bbox1.xmin() ||
        bbox2.ymin() > bbox1.ymax() || bbox2.ymax() < bbox1.ymin()) {
        // Return [0, 0, 0, 0] if there is no intersection.
        intersect_bbox->set_xmin(0);
        intersect_bbox->set_ymin(0);
        intersect_bbox->set_xmax(0);
        intersect_bbox->set_ymax(0);
    } else {
        intersect_bbox->set_xmin(std::max(bbox1.xmin(), bbox2.xmin()));
        intersect_bbox->set_ymin(std::max(bbox1.ymin(), bbox2.ymin()));
        intersect_bbox->set_xmax(std::min(bbox1.xmax(), bbox2.xmax()));
        intersect_bbox->set_ymax(std::min(bbox1.ymax(), bbox2.ymax()));
    }
}

float BBoxSize(const NormalizedBBox& bbox, const bool normalized = true) {
    if (bbox.xmax() < bbox.xmin() || bbox.ymax() < bbox.ymin()) {
        // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
        return 0;
    } else {
        if (bbox.has_size()) {
            return bbox.size();
        } else {
            float width = bbox.xmax() - bbox.xmin();
            float height = bbox.ymax() - bbox.ymin();
            if (normalized) {
                return width * height;
            } else {
                // If bbox is not within range [0, 1].
                return (width + 1) * (height + 1);
            }
        }
    }
}

Dtype BBoxSize(const Dtype* bbox, const bool normalized = true) {
    if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
        // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
        return Dtype(0.);
    } else {
        const Dtype width = bbox[2] - bbox[0];
        const Dtype height = bbox[3] - bbox[1];
        if (normalized) {
            return width * height;
        } else {
            // If bbox is not within range [0, 1].
            return (width + 1) * (height + 1);
        }
    }
}


void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clip_bbox) {
    clip_bbox->set_xmin(std::max(std::min(bbox.xmin(), 1.f), 0.f));
    clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), 1.f), 0.f));
    clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), 1.f), 0.f));
    clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), 1.f), 0.f));
    clip_bbox->clear_size();
    clip_bbox->set_size(BBoxSize(*clip_bbox));
    clip_bbox->set_difficult(bbox.difficult());
}

void ClipBBox(const NormalizedBBox& bbox, const float height, const float width,
              NormalizedBBox* clip_bbox) {
    clip_bbox->set_xmin(std::max(std::min(bbox.xmin(), width), 0.f));
    clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), height), 0.f));
    clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), width), 0.f));
    clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), height), 0.f));
    clip_bbox->clear_size();
    clip_bbox->set_size(BBoxSize(*clip_bbox));
    clip_bbox->set_difficult(bbox.difficult());
}

void GetLocPredictions(const Dtype* loc_data, const int num,
                       const int num_preds_per_class, const int num_loc_classes,
                       const bool share_location, std::vector<LabelBBox>* loc_preds) {
    loc_preds->clear();
    //num_loc_classes = 1;
    loc_preds->resize(num);
    for (int i = 0; i < num; ++i) {
        LabelBBox& label_bbox = (*loc_preds)[i];
        for (int p = 0; p < num_preds_per_class; ++p) {
            int start_idx = p * num_loc_classes * 4;
            for (int c = 0; c < num_loc_classes; ++c) {
                int label = share_location ? -1 : c;
                if (label_bbox.find(label) == label_bbox.end()) {
                    label_bbox[label].resize(num_preds_per_class);
                }
                label_bbox[label][p].set_xmin(loc_data[start_idx + c * 4]);
                label_bbox[label][p].set_ymin(loc_data[start_idx + c * 4 + 1]);
                label_bbox[label][p].set_xmax(loc_data[start_idx + c * 4 + 2]);
                label_bbox[label][p].set_ymax(loc_data[start_idx + c * 4 + 3]);
            }
        }
        loc_data += num_preds_per_class * num_loc_classes * 4;
    }
}

void GetConfidenceScores(const Dtype* conf_data, const int num,
                         const int num_preds_per_class, const int num_classes,
                         vector<map<int, vector<float> > >* conf_preds) {
conf_preds->clear();
conf_preds->resize(num);
for (int i = 0; i < num; ++i) {
map<int, vector<float> >& label_scores = (*conf_preds)[i];
for (int p = 0; p < num_preds_per_class; ++p) {
int start_idx = p * num_classes;
for (int c = 0; c < num_classes; ++c) {
label_scores[c].push_back(conf_data[start_idx + c]);
}
}
conf_data += num_preds_per_class * num_classes;
}
}

void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
                    vector<NormalizedBBox>* prior_bboxes,
                    vector<vector<float> >* prior_variances) {
    prior_bboxes->clear();
    prior_variances->clear();
    for (int i = 0; i < num_priors; ++i) {
        int start_idx = i * 4;
        NormalizedBBox bbox;
        bbox.set_xmin(prior_data[start_idx]);
        bbox.set_ymin(prior_data[start_idx + 1]);
        bbox.set_xmax(prior_data[start_idx + 2]);
        bbox.set_ymax(prior_data[start_idx + 3]);
        float bbox_size = BBoxSize(bbox);
        bbox.set_size(bbox_size);
        prior_bboxes->push_back(bbox);
    }

    for (int i = 0; i < num_priors; ++i) {
        int start_idx = (num_priors + i) * 4;
        vector<float> var;
        for (int j = 0; j < 4; ++j) {
            var.push_back(prior_data[start_idx + j]);
        }
        prior_variances->push_back(var);
    }
}

void DecodeBBox(
        const NormalizedBBox& prior_bbox, const vector<float>& prior_variance,
        const CodeType code_type, const bool variance_encoded_in_target,
        const bool clip_bbox, const NormalizedBBox& bbox,
        NormalizedBBox* decode_bbox) {
    if (code_type == PriorBoxParameter_CodeType_CORNER) {
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to add the offset
            // predictions.
            decode_bbox->set_xmin(prior_bbox.xmin() + bbox.xmin());
            decode_bbox->set_ymin(prior_bbox.ymin() + bbox.ymin());
            decode_bbox->set_xmax(prior_bbox.xmax() + bbox.xmax());
            decode_bbox->set_ymax(prior_bbox.ymax() + bbox.ymax());
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox->set_xmin(
                    prior_bbox.xmin() + prior_variance[0] * bbox.xmin());
            decode_bbox->set_ymin(
                    prior_bbox.ymin() + prior_variance[1] * bbox.ymin());
            decode_bbox->set_xmax(
                    prior_bbox.xmax() + prior_variance[2] * bbox.xmax());
            decode_bbox->set_ymax(
                    prior_bbox.ymax() + prior_variance[3] * bbox.ymax());
        }
    } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
        float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
        CHECK_GT(prior_width, 0);
        float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
        CHECK_GT(prior_height, 0);
        float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
        float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;

        float decode_bbox_center_x, decode_bbox_center_y;
        float decode_bbox_width, decode_bbox_height;
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to retore the offset
            // predictions.
            decode_bbox_center_x = bbox.xmin() * prior_width + prior_center_x;
            decode_bbox_center_y = bbox.ymin() * prior_height + prior_center_y;
            decode_bbox_width = exp(bbox.xmax()) * prior_width;
            decode_bbox_height = exp(bbox.ymax()) * prior_height;
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox_center_x =
                    prior_variance[0] * bbox.xmin() * prior_width + prior_center_x;
            decode_bbox_center_y =
                    prior_variance[1] * bbox.ymin() * prior_height + prior_center_y;
            decode_bbox_width =
                    exp(prior_variance[2] * bbox.xmax()) * prior_width;
            decode_bbox_height =
                    exp(prior_variance[3] * bbox.ymax()) * prior_height;
        }

        decode_bbox->set_xmin(decode_bbox_center_x - decode_bbox_width / 2.);
        decode_bbox->set_ymin(decode_bbox_center_y - decode_bbox_height / 2.);
        decode_bbox->set_xmax(decode_bbox_center_x + decode_bbox_width / 2.);
        decode_bbox->set_ymax(decode_bbox_center_y + decode_bbox_height / 2.);
    } else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
        float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
        CHECK_GT(prior_width, 0);
        float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
        CHECK_GT(prior_height, 0);
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to add the offset
            // predictions.
            decode_bbox->set_xmin(prior_bbox.xmin() + bbox.xmin() * prior_width);
            decode_bbox->set_ymin(prior_bbox.ymin() + bbox.ymin() * prior_height);
            decode_bbox->set_xmax(prior_bbox.xmax() + bbox.xmax() * prior_width);
            decode_bbox->set_ymax(prior_bbox.ymax() + bbox.ymax() * prior_height);
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox->set_xmin(
                    prior_bbox.xmin() + prior_variance[0] * bbox.xmin() * prior_width);
            decode_bbox->set_ymin(
                    prior_bbox.ymin() + prior_variance[1] * bbox.ymin() * prior_height);
            decode_bbox->set_xmax(
                    prior_bbox.xmax() + prior_variance[2] * bbox.xmax() * prior_width);
            decode_bbox->set_ymax(
                    prior_bbox.ymax() + prior_variance[3] * bbox.ymax() * prior_height);
        }
    } else {
        fprintf(stderr, "Unknown LocLossType.\n");
    }
    float bbox_size = BBoxSize(*decode_bbox);
    decode_bbox->set_size(bbox_size);
    if (clip_bbox) {
        ClipBBox(*decode_bbox, decode_bbox);
    }
}

void DecodeBBoxes(
        const vector<NormalizedBBox>& prior_bboxes,
        const vector<vector<float> >& prior_variances,
        const CodeType code_type, const bool variance_encoded_in_target,
        const bool clip_bbox, const vector<NormalizedBBox>& bboxes,
        vector<NormalizedBBox>* decode_bboxes) {
    CHECK_EQ(prior_bboxes.size(), prior_variances.size());
    CHECK_EQ(prior_bboxes.size(), bboxes.size());
    int num_bboxes = prior_bboxes.size();
    if (num_bboxes >= 1) {
        CHECK_EQ(prior_variances[0].size(), 4);
    }
    decode_bboxes->clear();
    for (int i = 0; i < num_bboxes; ++i) {
        NormalizedBBox decode_bbox;
        DecodeBBox(prior_bboxes[i], prior_variances[i], code_type,
                   variance_encoded_in_target, clip_bbox, bboxes[i], &decode_bbox);
        decode_bboxes->push_back(decode_bbox);
    }
}

void DecodeBBoxesAll(const vector<LabelBBox>& all_loc_preds,
                     const vector<NormalizedBBox>& prior_bboxes,
                     const vector<vector<float> >& prior_variances,
                     const int num, const bool share_location,
                     const int num_loc_classes, const int background_label_id,
                     const CodeType code_type, const bool variance_encoded_in_target,
                     const bool clip, vector<LabelBBox>* all_decode_bboxes) {
    //CHECK_EQ(all_loc_preds.size(), num);
    all_decode_bboxes->clear();
    all_decode_bboxes->resize(num);
    for (int i = 0; i < num; ++i) {
        // Decode predictions into bboxes.
        LabelBBox& decode_bboxes = (*all_decode_bboxes)[i];
        for (int c = 0; c < num_loc_classes; ++c) {
            int label = share_location ? -1 : c;
            if (label == background_label_id) {
                // Ignore background class.
                continue;
            }
            if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
                // Something bad happened if there are no predictions for current label.
                fprintf(stderr, "Could not find location predictions for label %d\n", label);
            }
            const vector<NormalizedBBox>& label_loc_preds =
                    all_loc_preds[i].find(label)->second;
            DecodeBBoxes(prior_bboxes, prior_variances,
                         code_type, variance_encoded_in_target, clip,
                         label_loc_preds, &(decode_bboxes[label]));
        }
    }
}

void GetMaxScoreIndex(const vector<float>& scores, const float threshold,
                      const int top_k, vector<pair<float, int> >* score_index_vec) {
    // Generate index score pairs.
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            score_index_vec->push_back(std::make_pair(scores[i], i));
        }
    }

    // Sort the score pair according to the scores in descending order
    std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                     SortScorePairDescend<int>);

    // Keep top_k scores if needed.
    if (top_k > -1 && top_k < score_index_vec->size()) {
        score_index_vec->resize(top_k);
    }
}

void GetMaxScoreIndex(const Dtype* scores, const int num, const float threshold,
                      const int top_k, vector<pair<Dtype, int> >* score_index_vec) {
    // Generate index score pairs.
    for (int i = 0; i < num; ++i) {
        if (scores[i] > threshold) {
            score_index_vec->push_back(std::make_pair(scores[i], i));
        }
    }

    // Sort the score pair according to the scores in descending order
    std::sort(score_index_vec->begin(), score_index_vec->end(),
              SortScorePairDescend<int>);

    // Keep top_k scores if needed.
    if (top_k > -1 && top_k < score_index_vec->size()) {
        score_index_vec->resize(top_k);
    }
}

float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                     const bool normalized = true) {
    NormalizedBBox intersect_bbox;
    IntersectBBox(bbox1, bbox2, &intersect_bbox);
    float intersect_width, intersect_height;
    if (normalized) {
        intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin();
        intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin();
    } else {
        intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin() + 1;
        intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin() + 1;
    }
    if (intersect_width > 0 && intersect_height > 0) {
        float intersect_size = intersect_width * intersect_height;
        float bbox1_size = BBoxSize(bbox1);
        float bbox2_size = BBoxSize(bbox2);
        return intersect_size / (bbox1_size + bbox2_size - intersect_size);
    } else {
        return 0.;
    }
}

Dtype JaccardOverlap(const Dtype* bbox1, const Dtype* bbox2) {
    if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
        bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
        return Dtype(0.);
    } else {
        const Dtype inter_xmin = std::max(bbox1[0], bbox2[0]);
        const Dtype inter_ymin = std::max(bbox1[1], bbox2[1]);
        const Dtype inter_xmax = std::min(bbox1[2], bbox2[2]);
        const Dtype inter_ymax = std::min(bbox1[3], bbox2[3]);

        const Dtype inter_width = inter_xmax - inter_xmin;
        const Dtype inter_height = inter_ymax - inter_ymin;
        const Dtype inter_size = inter_width * inter_height;

        const Dtype bbox1_size = BBoxSize(bbox1);
        const Dtype bbox2_size = BBoxSize(bbox2);

        return inter_size / (bbox1_size + bbox2_size - inter_size);
    }
}

void ApplyNMSFast(const vector<NormalizedBBox>& bboxes,
                  const vector<float>& scores, const float score_threshold,
                  const float nms_threshold, const float eta, const int top_k,
                  vector<int>* indices) {
    // Sanity check.
    if(!CHECK_EQ(bboxes.size(), scores.size()))
        fprintf(stderr, "bboxes and scores have different size.\n");

    // Get top_k scores (with corresponding indices).
    vector<pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices->clear();
    while (score_index_vec.size() != 0) {
        const int idx = score_index_vec.front().second;
        bool keep = true;
        for (int k = 0; k < indices->size(); ++k) {
            if (keep) {
                const int kept_idx = (*indices)[k];
                float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
                keep = overlap <= adaptive_threshold;
            } else {
                break;
            }
        }
        if (keep) {
            indices->push_back(idx);
        }
        score_index_vec.erase(score_index_vec.begin());
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
            adaptive_threshold *= eta;
        }
    }
}

void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
                  const float score_threshold, const float nms_threshold,
                  const float eta, const int top_k, vector<int>* indices) {
    // Get top_k scores (with corresponding indices).
    vector<pair<Dtype, int> > score_index_vec;
    GetMaxScoreIndex(scores, num, score_threshold, top_k, &score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices->clear();
    while (score_index_vec.size() != 0) {
        const int idx = score_index_vec.front().second;
        bool keep = true;
        for (int k = 0; k < indices->size(); ++k) {
            if (keep) {
                const int kept_idx = (*indices)[k];
                float overlap = JaccardOverlap(bboxes + idx * 4, bboxes + kept_idx * 4);
                keep = overlap <= adaptive_threshold;
            } else {
                break;
            }
        }
        if (keep) {
            indices->push_back(idx);
        }
        score_index_vec.erase(score_index_vec.begin());
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
            adaptive_threshold *= eta;
        }
    }
}

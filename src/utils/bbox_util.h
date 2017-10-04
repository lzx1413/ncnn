//
// Created by fotoable on 2017/9/27.
//

#ifndef NCNN_BBOX_UTIL_H
#define NCNN_BBOX_UTIL_H
#include <math.h>
#include <algorithm>
#include <map>
#include <vector>
//#include "caffe.pb.h"
#include "NormalizedBBox.h"
using namespace std;
class Rect
{
public:
float x, y, width, height;

Rect() : x(0.f), y(0.f), width(0.f), height(0.f) {}
Rect(float _x, float _y, float _width, float _height) :x(_x), y(_y), width(_width), height(_height) {}
float area() const { return width * height; }
float inter_area(const Rect& rhs) const
{
    float x2 = x + width;
    float y2 = y + height;
    float rhs_x2 = rhs.x + rhs.width;
    float rhs_y2 = rhs.y + rhs.height;

    float xL = std::max(x, rhs.x);
    float xR = std::min(x2, rhs_x2);
    if (xR <= xL)
        return 0.f;

    float yT = std::max(y, rhs.y);
    float yB = std::min(y2, rhs_y2);
    if (yB <= yT)
        return 0.f;

    return (xR - xL) * (yB - yT);
}
};

class ProposalBox
{
public:
    Rect box;
    float score;
    float area() const { return box.area(); }
    float inter_area(const ProposalBox& rhs) const { return box.inter_area(rhs.box); }
    bool operator<(const ProposalBox& rhs) const { return score > rhs.score; }
};

std::vector<int> nms(const std::vector<ProposalBox>& boxes, float nms_thresh);

//ssd bbox functions
enum PriorBoxParameter_CodeType {
    PriorBoxParameter_CodeType_CORNER = 1,
    PriorBoxParameter_CodeType_CENTER_SIZE = 2,
    PriorBoxParameter_CodeType_CORNER_SIZE = 3
};
typedef PriorBoxParameter_CodeType CodeType;
typedef map<int, std::vector<NormalizedBBox> > LabelBBox;
typedef float Dtype;
void GetLocPredictions(const Dtype* loc_data, const int num,
                       const int num_preds_per_class, const int num_loc_classes,
                       const bool share_location, std::vector<LabelBBox>* loc_preds);

void GetConfidenceScores(const Dtype* conf_data, const int num,
                         const int num_preds_per_class, const int num_classes,
                         vector<map<int, vector<float> > >* conf_preds);
void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
                    vector<NormalizedBBox>* prior_bboxes,
                    vector<vector<float> >* prior_variances);
void DecodeBBoxesAll(const vector<LabelBBox>& all_loc_preds,
                     const vector<NormalizedBBox>& prior_bboxes,
                     const vector<vector<float> >& prior_variances,
                     const int num, const bool share_location,
                     const int num_loc_classes, const int background_label_id,
                     const CodeType code_type, const bool variance_encoded_in_target,
                     const bool clip, vector<LabelBBox>* all_decode_bboxes);
void ApplyNMSFast(const vector<NormalizedBBox>& bboxes,
                  const vector<float>& scores, const float score_threshold,
                  const float nms_threshold, const float eta, const int top_k,
                  vector<int>* indices);
void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
                  const float score_threshold, const float nms_threshold,
                  const float eta, const int top_k, vector<int>* indices);
template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2) {
    return pair1.first > pair2.first;
}

// Explicit initialization.
template bool SortScorePairDescend(const pair<float, int>& pair1,
                                   const pair<float, int>& pair2);
template bool SortScorePairDescend(const pair<float, pair<int, int> >& pair1,
                                   const pair<float, pair<int, int> >& pair2);



#endif //NCNN_BBOX_UTIL_H

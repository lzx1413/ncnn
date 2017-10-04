//
// Created by fotoable on 2017/9/27.
//

#ifndef NCNN_BBOX_UTIL_H
#define NCNN_BBOX_UTIL_H
#include <math.h>
#include <algorithm>
#include <vector>
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




#endif //NCNN_BBOX_UTIL_H

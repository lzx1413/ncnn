//
// Created by lzx1413 on 2017/9/27.
//

#include "bbox_util.h"
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
